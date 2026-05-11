from dataclasses import dataclass
from typing import Any

from backend.services.dto.analysis import BaseEvidence


@dataclass(frozen=True)
class RoutingDecision:
    use_llm: bool
    route_reason: str


class ConfidenceRouter:
    def decide(self, evidence: BaseEvidence, base_result: dict[str, Any]) -> RoutingDecision:
        model_label = evidence.model_label
        model_confidence = float(evidence.model_confidence or 0.0)
        model_margin = float(evidence.model_margin or 0.0)
        rule_score = evidence.rule_score
        rule_hits = evidence.rule_hits
        chunk_consistent = evidence.chunk_consistent

        if not chunk_consistent:
            return RoutingDecision(True, "chunk_predictions_conflict")
        if model_label == "中等風險" and model_confidence < 0.7:
            return RoutingDecision(True, "model_medium")
        if model_margin < 0.12:
            return RoutingDecision(True, "low_margin")
        if model_label == "低風險" and rule_score >= 45:
            return RoutingDecision(True, "rules_high_but_model_low")
        if model_label == "高風險" and rule_score <= 20:
            return RoutingDecision(True, "model_high_but_rules_low")
        if model_label == "低風險" and model_confidence < 0.7 and len(rule_hits) >= 2:
            return RoutingDecision(True, "low_confidence_with_rule_hits")
        if base_result.get("label") == "中等風險":
            return RoutingDecision(True, "base_result_medium")
        return RoutingDecision(False, "base_confident")


confidence_router = ConfidenceRouter()
