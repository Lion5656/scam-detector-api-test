"""整合商品資訊、價格規則、黑名單與選用的 LLM 風險分析。"""

import json
import os
import re
from typing import Any, cast

from pydantic import SecretStr

from backend.config import settings
from backend.services.dto.price_analysis import (
    DecisionLayer,
    MarketPriceSource,
    RiskLabel,
)


class FusionDecisionEngine:
    """整合價格、黑名單與 Groq 產生風險決策。"""

    @staticmethod
    def _has_price_risk(selling_price: int, market_price: int) -> bool:
        """判斷正數售價是否低於市價五成，或達到市價兩倍以上。"""
        return (
            selling_price > 0
            and market_price > 0
            and (
                selling_price < market_price * 0.5
                or selling_price >= market_price * 2
            )
        )

    @staticmethod
    def _enforce_price_risk(
        decision: dict[str, Any],
        selling_price: int,
        market_price: int,
        has_price_risk: bool,
    ) -> dict[str, Any]:
        """以硬性價格規則修正最終決策，避免深度分析覆蓋價格異常。"""
        if not has_price_risk:
            return decision

        score = decision.get("risk_score")
        enforced_score = (
            max(float(score), 90.0)
            if isinstance(score, (int, float))
            else 90.0
        )
        reason = str(decision.get("reason") or "")
        if selling_price < market_price * 0.5:
            price_reason = (
                f"販售價格 {selling_price} 低於正常市價 {market_price} 的 50%，"
                "判定為高風險低於行情"
            )
            price_evidence = "低於行情 50% 規則觸發"
        else:
            price_reason = (
                f"販售價格 {selling_price} 達正常市價 {market_price} 的 2 倍以上，"
                "判定為高風險高於行情"
            )
            price_evidence = "高於行情 2 倍規則觸發"

        evidence = [str(item) for item in decision.get("evidence") or []]
        if price_evidence not in evidence:
            evidence.append(price_evidence)

        return {
            **decision,
            "risk_label": "高風險",
            "risk_score": enforced_score,
            "reason": f"{reason}；{price_reason}" if reason else price_reason,
            "evidence": evidence,
        }

    @staticmethod
    def _normalize_risk_label(label: object) -> RiskLabel:
        """將中英文風險標籤轉成服務使用的標準值。"""
        value = str(label or "").strip()
        normalized = value.upper()
        if normalized == "HIGH" or "高" in value:
            return "HIGH"
        if normalized == "MEDIUM" or "中" in value:
            return "MEDIUM"
        if normalized == "LOW" or "低" in value:
            return "LOW"
        return "UNKNOWN"

    @staticmethod
    def _normalize_decision_layer(layer: object) -> DecisionLayer:
        """驗證決策層名稱，未知值一律降級為快速決策層。"""
        value = str(layer or "fast").strip().lower()
        if value in {"fast", "llm", "llm_simulated", "source_validation"}:
            return cast(DecisionLayer, value)
        return "fast"

    def _load_blacklist_terms(self) -> tuple[list[str], list[str]]:
        """載入黑名單關鍵字與正則表達式；讀取失敗時回傳空清單。"""
        try:
            with open(settings.BLACKLIST_TERMS_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            keywords = [str(k) for k in payload.get("keywords", [])]
            patterns = [str(p) for p in payload.get("patterns", [])]
            return keywords, patterns
        except Exception:
            return [], []

    def _run_blacklist_hit(self, text: str) -> int:
        """計算 OCR 文字命中的黑名單關鍵字與有效正則表達式數量。"""
        keywords, patterns = self._load_blacklist_terms()
        hit_keywords = [k for k in keywords if k and k in text]

        hit_patterns: list[str] = []
        for p in patterns:
            try:
                if re.search(p, text, flags=re.IGNORECASE):
                    hit_patterns.append(p)
            except re.error:
                continue

        return len(hit_keywords) + len(hit_patterns)

    @staticmethod
    def _requires_deep_analysis(
        product_name: str,
        brand_model: str,
        selling_price: int,
        market_price: int,
        has_price_risk: bool,
        tools: dict[str, Any],
    ) -> bool:
        """依商品資料完整度、價格異常與黑名單命中決定是否深入分析。"""
        product_unknown = (
            not product_name
            or "未知" in product_name
            or not brand_model
            or "未知" in brand_model
            or "待人工" in brand_model
        )
        return (
            product_unknown
            or selling_price <= 0
            or market_price <= 0
            or has_price_risk
            or int(tools["blacklist"]) > 0
        )

    def _alt_deep_result(self, base_result: dict[str, Any], tools: dict[str, Any]) -> dict[str, Any]:
        """在 Groq 不可用時建立本地替代結果。"""
        base_score = base_result.get("risk_score", 0.0)

        risk_bonus = 0.0
        evidence: list[str] = []

        blacklist_hit = int(tools["blacklist"])
        if blacklist_hit > 0:
            risk_bonus += min(20.0, blacklist_hit * 6.0)
            evidence.append(f"黑名單命中 {blacklist_hit} 項")

        market_price_info = tools["market_price"]
        market_price = int(market_price_info.get("price", 0.0))
        market_price_source = str(
            market_price_info.get("source", "not_evaluated")
        )
        if market_price > 0 and market_price_source == "online":
            evidence.append(f"線上比價參考 {market_price}")
        elif market_price > 0 and market_price_source == "fallback_local":
            evidence.append(f"本地價格參考 {market_price}")

        score = min(
            100.0,
            max(5.0, base_score, base_score * 0.60 + risk_bonus * 0.40),
        )

        label = "低風險"
        if score >= 80:
            label = "高風險"
        elif score >= 40:
            label = "中等風險"

        reason = str(base_result.get("reason") or "")
        if evidence:
            reason = (reason + "；含以下風險" if reason else "") + "、".join(evidence)
        else:
            if base_result.get("has_risk") == "含價格風險":
                reason = reason + "，判定為高風險"
            else:
                reason = "價格判定屬於正常區間，判定為低風險"

        return {
            "risk_label": label,
            "risk_score": round(score, 2),
            "reason": reason or "依據多工具檢查完成風險評估",
            "evidence": evidence,
            "confidence": round(min(0.98, 0.6 + risk_bonus / 100.0), 2),
            "decision_layer": "llm_simulated",
            "market_price_source": market_price_source,
        }

    def _call_llm_deep_analysis(self, product_context: dict[str, Any], tools: dict[str, Any]) -> dict[str, Any] | None:
        """呼叫 Groq 產生 JSON 決策；設定缺失或回應無效時回傳空值。"""
        api_key = settings.GROQ_API_KEY or SecretStr(str(os.getenv("GROQ_API_KEY")))
        if not api_key:
            return None

        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_groq import ChatGroq
        except Exception:
            return None

        prompt = ChatPromptTemplate.from_template(
            """
            你是防詐 API 的深度決策器。請根據輸入資料，僅輸出 JSON。
            必要欄位：risk_label, risk_score, reason, evidence, confidence, decision_layer。

            [product_context]
            {product_context}

            [tool_outputs]
            {tool_outputs}

            規則：
            1) risk_label 只能是 低風險/中等風險/高風險/未知風險
            2) risk_score 範圍 0~100
            3) evidence 為字串陣列
            4) decision_layer 固定填 llm
            """
        )

        llm = ChatGroq(model=settings.RAG_MODEL_NAME, temperature=0.1, api_key=api_key)
        chain = prompt | llm
        try:
            raw = chain.invoke(
                {
                    "product_context": json.dumps(product_context, ensure_ascii=False),
                    "tool_outputs": json.dumps(tools, ensure_ascii=False),
                }
            )
            content = getattr(raw, "content", "")
            if not content:
                return None
            payload = json.loads(content)
            if not isinstance(payload, dict):
                return None
            payload.setdefault("risk_label", "UNKNOWN")
            payload.setdefault("risk_score", 0.0)
            payload.setdefault("decision_layer", "llm")
            payload.setdefault("evidence", [])
            payload.setdefault("confidence", 0.7)
            return payload
        except Exception:
            return None

    def evaluate(
        self,
        *,
        product_name: str,
        brand_model: str,
        text: str,
        selling_price: int,
        market_price: int,
        market_price_source: MarketPriceSource,
    ) -> dict[str, Any]:
        """整合商品資料並回傳最終風險結果。"""
        has_price_risk = self._has_price_risk(selling_price, market_price)
        tools = {
            "blacklist": self._run_blacklist_hit(text),
            "market_price": {
                "query": brand_model if brand_model and "未知" not in brand_model else product_name,
                "price": market_price,
                "source": market_price_source,
            },
        }

        base_result: dict[str, str | float] = {
            "has_risk": (
                "含價格風險"
                if has_price_risk
                else "尚未發現價格風險"
            ),
            "market_price_source": market_price_source,
            "risk_score": 90.0 if has_price_risk else 20.0,
            "reason": (
                "商品售價脫離正常範圍"
                if has_price_risk
                else "商品價格正常"
            ),
        }

        needs_deep_analysis = self._requires_deep_analysis(
            product_name,
            brand_model,
            selling_price,
            market_price,
            has_price_risk,
            tools,
        )
        if not needs_deep_analysis:
            decision: dict[str, Any] = {
                **base_result,
                "risk_label": (
                    "高風險"
                    if has_price_risk
                    else "低風險"
                ),
                "evidence": ["商品資訊與價格資料完整"],
                "confidence": 0.9,
                "decision_layer": "fast",
            }
        else:
            product_context = {
                "product_name": product_name,
                "brand_model": brand_model,
                "ocr_text": text,
                "selling_price": selling_price,
                "market_price": market_price,
                "market_price_source": market_price_source,
                "has_risk": has_price_risk,
                "base_result": base_result,
            }
            llm_result = self._call_llm_deep_analysis(product_context, tools)
            if llm_result:
                llm_result["market_price_source"] = market_price_source
                decision = llm_result
            else:
                decision = self._alt_deep_result(base_result, tools)

        decision = self._enforce_price_risk(
            decision,
            selling_price,
            market_price,
            has_price_risk,
        )
        return {
            **decision,
            "risk_label": self._normalize_risk_label(
                decision.get("risk_label")
            ),
            "decision_layer": self._normalize_decision_layer(
                decision.get("decision_layer")
            ),
        }


fusion_decision_engine = FusionDecisionEngine()
