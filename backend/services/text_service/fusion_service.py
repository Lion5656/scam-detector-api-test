from typing import Any

from backend.services.dto.analysis import RagEvidence


class FusionService:
    def merge(self, base_result: dict[str, Any],
        rag_result: RagEvidence,  route_reason: str) -> dict[str, Any]:

        merged = dict(base_result)
        merged["decision_source"] = "rag"

        rag_label = rag_result.label
        rag_score = rag_result.score
        rag_reason = rag_result.reason

        merged["llm_used"] = True
        merged["route_reason"] = route_reason

        merged["label"] = rag_label
        merged["score"] = f"{rag_score:.2f}" 
        merged["reason"] = rag_reason
        return merged


fusion_service = FusionService()
