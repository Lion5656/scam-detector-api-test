"""整合商品資訊、價格規則、黑名單與選用的 LLM 風險分析。"""

import json
import os
import re
from typing import Any

from pydantic import SecretStr

from backend.config import settings


class FusionDecisionEngine:
    """依商品資料與可用分析工具產生最終風險決策。

    服務會檢查 OCR 文字是否命中黑名單。當商品身分未知、價格資料不足、價格
    異常或黑名單命中時，才嘗試呼叫 Groq 語言模型；模型不可用或回應無效時，
    改用本地加權規則產生結果。
    """

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
        is_high_risk: bool,
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
            or is_high_risk
            or int(tools["blacklist"].get("hit_count") or 0) > 0
        )

    def _alt_deep_result(self, base_result: dict[str, Any], tools: dict[str, Any]) -> dict[str, Any]:
        """在 LLM 不可用時，以基礎分數和黑名單加權值建立替代結果。

        每筆黑名單命中提供 6 點加權值，上限為 20；最終分數由 60% 基礎分數與
        40% 加權值組成，並限制在 5 至 100。80 分以上為高風險，40 分以上為
        中等風險，其餘為低風險；線上市場價只會加入判定證據。
        """
        base_score = base_result.get("risk_score", 0.0)

        risk_bonus = 0.0
        evidence: list[str] = []

        blacklist_hit = int(tools["blacklist"])
        if blacklist_hit > 0:
            risk_bonus += min(20.0, blacklist_hit * 6.0)
            evidence.append(f"黑名單命中 {blacklist_hit} 項")

        online_price = int(tools["online_price"].get("price", 0.0))
        if online_price > 0:
            evidence.append(f"線上比價參考 {online_price}")

        score = min(100.0, max(5.0, base_score * 0.60 + risk_bonus * 0.40))

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
                reason = "未檢測到明顯特徵，判定為低風險"

        return {
            "risk_label": label,
            "risk_score": round(score, 2),
            "reason": reason or "依據多工具檢查完成風險評估",
            "evidence": evidence,
            "confidence": round(min(0.98, 0.6 + risk_bonus / 100.0), 2),
            "decision_layer": "alt"
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
        has_risk: bool,
    ) -> dict[str, Any]:
        """整合商品資料、黑名單、市場價、價格規則與選用的 LLM。

        回傳資料包含風險標籤、分數、原因、證據、信心分數與選用的決策層；
        此流程不接收一般文字分類器的結果。
        """
        tools = {
            "blacklist": self._run_blacklist_hit(text),
            "online_price": {
                "query": brand_model if brand_model and "未知" not in brand_model else product_name,
                "price": market_price
            },
        }

        base_result: dict[str, str | float] = {
            "has_risk": "含價格風險" if has_risk else "尚未發現價格風險",
            "risk_score": 90.0 if has_risk else 20.0,
            "reason": "商品售價脫離正常範圍" if has_risk else "商品價格正常"
            ,
        }

        needs_deep_analysis = self._requires_deep_analysis(
            product_name,
            brand_model,
            selling_price,
            market_price,
            has_risk,
            tools,
        )
        if not needs_deep_analysis:
            return {
                **base_result,
                "evidence": ["商品資訊與價格資料完整"],
                "confidence": 0.9
            }

        product_context = {
            "product_name": product_name,
            "brand_model": brand_model,
            "ocr_text": text,
            "selling_price": selling_price,
            "market_price": market_price,
            "has_risk": has_risk,
            "base_result": base_result,
        }
        llm_result = self._call_llm_deep_analysis(product_context, tools)
        if llm_result:
            return llm_result

        alt_result = self._alt_deep_result(base_result, tools)
        return alt_result


fusion_decision_engine = FusionDecisionEngine()
