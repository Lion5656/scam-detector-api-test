from typing import Any, cast

from backend.core.config import settings
from backend.rag.rag_reasoner import analyze_with_rag
from backend.services.dto.text_analysis import BaseEvidence
from backend.services.text_service.base_classifier import base_classifier
from backend.services.text_service.confidence_router import confidence_router
from backend.services.text_service.fusion import fusion
from backend.utils.pattern.text import compare_rules
from backend.utils.text_cleaner import normalize_text


class TextAnalyzer:
    """文字分析協調層 - 負責功能編排與決策路由"""

    def _build_base_evidence(self, text: str) -> BaseEvidence:
        """構建基礎證據：規則特徵 + 模型預測"""
        cleaned = normalize_text(text)
        rule_score, rule_reason, rule_hits = compare_rules(cleaned)
        model_result = base_classifier.predict_text(cleaned)
        return BaseEvidence(
            text=cleaned,
            rule_score=rule_score,
            rule_reason=rule_reason,
            rule_hits=list(set(rule_hits)),
            model_label=cast(str, model_result["label"]),
            model_confidence=cast(float | None, model_result["confidence"]),
            model_margin=cast(float | None, model_result["margin"]),
            chunk_consistent=cast(bool, model_result["chunk_consistent"]),
        )

    def _base_decision(self, evidence: BaseEvidence) -> dict[str, Any]:
        """基礎決策：融合規則和模型預測"""
        w_model = settings.MODEL_WEIGHT
        w_rule = settings.REGEX_WEIGHT

        rule_score = evidence.rule_score
        rule_reason = evidence.rule_reason
        rule_hits = set(evidence.rule_hits)
        model_label = evidence.model_label
        model_confidence = float(evidence.model_confidence or 0.0)
        extra_reason = rule_reason if "反詐" in rule_reason else ""

        response: dict[str, Any] = {
            "label": "",
            "score": "",
            "cls_model_confidence": model_confidence,
            "reason": "",
        }

        if model_label == "未知風險" and not rule_hits:
            response.update({"label": "未知風險", "score": "未知", "reason": "語句缺乏明確資訊，無法進行有效判斷，評估風險為未知"})
            return response
        if model_label == "低風險" and model_confidence >= 0.8 and len(rule_hits) <= 2:
            response.update({"label": "低風險", "score": 10.0, "reason": f"此訊息所含詐騙特徵較少{extra_reason}，評估風險為低。"})
            return response
        if model_label == "高風險" and (rule_score >= 45 or len(rule_hits) >= 2):
            response.update({"label": "高風險", "score": 85.0, "reason": f"此訊息{rule_reason}，評估風險為高"})
            return response
        if rule_score >= 85 and len(rule_hits) >= 3:
            response.update({"label": "高風險", "score": 95.0, "reason": f"此訊息{rule_reason}，評估風險為高"})
            return response
        if rule_score <= 20 and len(rule_hits) <= 2:
            response.update({"label": "低風險", "score": 20.0, "reason": "此訊息所含詐騙特徵較少，評估風險為低。"})
            return response

        if model_label == "高風險":
            model_eval = 100
        elif model_label == "中等風險":
            model_eval = 50
        elif model_label == "低風險":
            model_eval = 20
        else:
            model_eval = -1

        final_score = (20 + rule_score) * w_rule + model_eval * w_model

        if model_label == "低風險" and rule_score <= 28:
            final_score = min(final_score, 30.0)

        if final_score >= 80:
            response.update({"label": "高風險", "score": final_score, "reason": f"此訊息{rule_reason}，評估風險為高"})
            return response
        if final_score >= 40 and rule_reason:
            response.update({"label": "中等風險", "score": final_score, "reason": f"此訊息{rule_reason}，評估風險為中等"})
            return response
        if final_score >= 40:
            response.update({"label": "中等風險", "score": final_score, "reason": "此訊息疑似有詐騙風險，但特徵較模糊，評估風險為中等"})
            return response
        if model_eval < 0:
            response.update({"label": "未知風險", "score": "未知", "reason": "此訊息缺乏明確資訊，無法進行有效判斷，評估風險為未知"})
            return response
        response.update({"label": "低風險", "score": final_score, "reason": f"此訊息所含詐騙特徵較少{extra_reason}，評估風險為低。"})
        return response

    async def hybrid_detector(self, text: str) -> dict[str, Any]:
        """混合檢測：規則 + 模型 + 可選的 LLM/RAG"""
        evidence = self._build_base_evidence(text)
        base_result = self._base_decision(evidence)
        decision = confidence_router.decide(evidence, base_result)
        if not decision.use_llm:
            base_result["decision_source"] = "base"
            base_result["llm_used"] = False
            base_result["route_reason"] = decision.route_reason
            return base_result

        try:
            rag_result = await analyze_with_rag(evidence.text)
        except Exception as exc:
            fallback = dict(base_result)
            fallback["decision_source"] = "base"
            fallback["llm_used"] = False
            fallback["route_reason"] = f"rag_failed:{exc.__class__.__name__}"
            return fallback

        return fusion.merge(base_result, rag_result, decision.route_reason)

    def model_detector(self, text: str) -> dict[str, Any]:
        """純模型檢測"""
        prediction = base_classifier.predict_text(text)
        return {
            "label": prediction["label"] or "",
            "cls_model_confidence": float(prediction.get("confidence") or 0.0),
        }


# 全局實例
text_analyzer = TextAnalyzer()
