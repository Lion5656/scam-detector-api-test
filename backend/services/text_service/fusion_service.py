from typing import Any

from backend.services.dto.analysis import RagEvidence


class FusionService:
    def merge(
        self,
        base_result: dict[str, Any],
        rag_result: RagEvidence,
        route_reason: str,
    ) -> dict[str, Any]:
        merged = dict(base_result)
        merged["decision_source"] = "base"
        merged["llm_used"] = False
        merged["route_reason"] = route_reason

        if not rag_result.used:
            return merged

        rag_label = rag_result.label
        rag_score = rag_result.score
        rag_reason = rag_result.reason

        merged["llm_used"] = True
        merged["route_reason"] = route_reason

        if rag_label == base_result.get("label"):
            merged["decision_source"] = "hybrid"
            merged["score"] = rag_score if rag_score is not None else base_result.get("score")
            merged["reason"] = rag_reason or base_result.get("reason")
            return merged

        if base_result.get("label") == "中等風險":
            merged["label"] = rag_label or base_result.get("label")
            merged["score"] = rag_score if rag_score is not None else base_result.get("score")
            merged["reason"] = rag_reason or base_result.get("reason")
            merged["decision_source"] = "rag"
            return merged

        if base_result.get("label") == "未知風險":
            merged["label"] = rag_label or "未知風險"
            merged["score"] = rag_score if rag_score is not None else base_result.get("score")
            merged["reason"] = rag_reason or base_result.get("reason")
            merged["decision_source"] = "rag"
            return merged

        if base_result.get("label") == "低風險" and rag_label in {"中等風險", "高風險"}:
            merged["label"] = rag_label
            merged["score"] = rag_score if rag_score is not None else 55.0
            merged["reason"] = rag_reason or base_result.get("reason")
            merged["decision_source"] = "hybrid"
            return merged

        if base_result.get("label") == "高風險" and rag_label == "低風險":
            merged["label"] = "中等風險"
            merged["score"] = rag_score if rag_score is not None else 50.0
            merged["reason"] = rag_reason or base_result.get("reason")
            merged["decision_source"] = "hybrid"
            return merged

        merged["reason"] = rag_reason or base_result.get("reason")
        merged["decision_source"] = "hybrid"
        return merged


fusion_service = FusionService()
