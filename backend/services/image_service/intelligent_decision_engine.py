import json
import os
import re
from typing import Any

from backend.config import settings
from backend.rag.rag_retriever import retrieve_similar_cases
from backend.services.image_service.online_market_price_service import OnlineMarketPriceService


class IntelligentDecisionEngine:
    def __init__(self):
        self._online_price_service = OnlineMarketPriceService()

    def _load_blacklist_terms(self) -> tuple[list[str], list[str]]:
        try:
            with open(settings.BLACKLIST_TERMS_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            keywords = [str(k) for k in payload.get("keywords", [])]
            patterns = [str(p) for p in payload.get("patterns", [])]
            return keywords, patterns
        except Exception:
            return [], []

    def _run_blacklist_tool(self, text: str) -> dict[str, Any]:
        keywords, patterns = self._load_blacklist_terms()
        hit_keywords = [k for k in keywords if k and k in text]

        hit_patterns: list[str] = []
        for p in patterns:
            try:
                if re.search(p, text, flags=re.IGNORECASE):
                    hit_patterns.append(p)
            except re.error:
                continue

        return {
            "hit_keywords": hit_keywords,
            "hit_patterns": hit_patterns,
            "hit_count": len(hit_keywords) + len(hit_patterns),
        }

    def _run_rag_tool(self, text: str) -> dict[str, Any]:
        try:
            docs = retrieve_similar_cases(text)
            return {"count": len(docs)}
        except Exception:
            return {"count": 0}

    def _sanitize_tool_observations(self, tools: dict[str, Any]) -> dict[str, Any]:
        blacklist = tools.get("blacklist") or {}
        rag = tools.get("rag") or {}
        online = tools.get("online_price") or {}

        return {
            "blacklist_hit_count": int(blacklist.get("hit_count") or 0),
            "rag_case_count": int(rag.get("count") or 0),
            "online_price_query": str(online.get("query") or ""),
            "online_price": int(online.get("price") or 0),
        }

    def _run_online_price_tool(self, brand_model: str, product_name: str) -> dict[str, Any]:
        query = brand_model if brand_model and "未知" not in brand_model else product_name
        if not query:
            return {"query": "", "price": 0}

        price = self._online_price_service.estimate_taiwan_market_price(
            query,
            max_results=settings.ONLINE_PRICE_MAX_RESULTS,
        )
        return {"query": query, "price": price}

    def _is_uncertain(self, text_result: dict[str, Any], product_name: str, brand_model: str) -> bool:
        label = str(text_result.get("label", ""))
        model_conf = float(text_result.get("cls_model_confidence") or 0.0)
        if label == "中等風險":
            return True
        if model_conf < settings.INTELLIGENT_CONFIDENCE_THRESHOLD:
            return True
        if "未知" in product_name or "未知" in brand_model or "待人工" in brand_model:
            return True
        return False

    def _heuristic_deep_result(self, text_result: dict[str, Any], tools: dict[str, Any]) -> dict[str, Any]:
        base_label = str(text_result.get("label", "未知風險"))
        base_score = text_result.get("score")
        try:
            base_score_num = float(base_score)
        except (TypeError, ValueError):
            base_score_num = 50.0 if base_label == "中等風險" else 25.0

        risk_bonus = 0.0
        evidence: list[str] = []

        blacklist_hit = int(tools["blacklist"]["hit_count"])
        if blacklist_hit > 0:
            risk_bonus += min(20.0, blacklist_hit * 6.0)
            evidence.append(f"黑名單命中 {blacklist_hit} 項")

        if int(tools["rag"]["count"]) > 0:
            evidence.append("檢索到相似案例")

        online_price = int(tools["online_price"].get("price") or 0)
        if online_price > 0:
            evidence.append(f"線上比價參考 {online_price}")

        score = min(100.0, max(5.0, base_score_num + risk_bonus))
        label = "低風險"
        if score >= 80:
            label = "高風險"
        elif score >= 40:
            label = "中等風險"

        reason = str(text_result.get("reason") or "")
        if evidence:
            reason = (reason + "；" if reason else "") + "、".join(evidence)

        return {
            "risk_label": label,
            "risk_score": round(score, 2),
            "reason": reason or "依據多工具檢查完成風險評估",
            "evidence": evidence,
            "confidence": round(min(0.98, 0.6 + risk_bonus / 100.0), 2),
            "decision_layer": "llm_simulated",
        }

    def _call_llm_deep_analysis(self, text_result: dict[str, Any], tools: dict[str, Any]) -> dict[str, Any] | None:
        api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not api_key:
            return None

        try:
            from langchain_groq import ChatGroq
            from langchain_core.prompts import ChatPromptTemplate
        except Exception:
            return None

        prompt = ChatPromptTemplate.from_template(
            """
你是防詐 API 的深度決策器。請根據輸入資料，僅輸出 JSON。
必要欄位：risk_label, risk_score, reason, evidence, confidence, decision_layer。

[base_result]
{base_result}

[tool_outputs]
{tool_outputs}

規則：
1) risk_label 只能是 低風險/中等風險/高風險/未知風險
2) risk_score 範圍 0~100
3) evidence 為字串陣列
4) decision_layer 固定填 llm
"""
        )

        llm = ChatGroq(model_name=settings.RAG_MODEL_NAME, temperature=0.1, api_key=api_key)
        chain = prompt | llm
        try:
            raw = chain.invoke(
                {
                    "base_result": json.dumps(text_result, ensure_ascii=False),
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

    def evaluate(self, text_result: dict[str, Any], product_name: str, brand_model: str, text: str) -> dict[str, Any]:
        uncertain = self._is_uncertain(text_result, product_name, brand_model)
        if not uncertain:
            return {
                "risk_label": str(text_result.get("label", "未知風險")),
                "risk_score": text_result.get("score"),
                "reason": str(text_result.get("reason") or ""),
                "evidence": ["第一層快篩信心充足"],
                "confidence": float(text_result.get("cls_model_confidence") or 0.9),
                "decision_layer": "fast",
                "tool_observations": {},
            }

        tools = {
            "blacklist": self._run_blacklist_tool(text),
            "rag": self._run_rag_tool(text),
            "online_price": self._run_online_price_tool(brand_model, product_name),
        }

        llm_result = self._call_llm_deep_analysis(text_result, tools)
        if llm_result:
            llm_result["tool_observations"] = self._sanitize_tool_observations(tools)
            return llm_result

        result = self._heuristic_deep_result(text_result, tools)
        result["tool_observations"] = self._sanitize_tool_observations(tools)
        return result


intelligent_decision_engine = IntelligentDecisionEngine()
