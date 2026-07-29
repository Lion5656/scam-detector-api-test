"""依市場價格區間、商品狀態及選用的狀態複核產生風險決策。"""

import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import ValidationError

from backend.services.dto.price_analysis import (
    DeepAnalysisReview,
    MarketPriceEstimate,
    MarketPriceSource,
    RiskLabel,
)
from backend.services.image_price_service.domain.models import MarketplaceCondition
from backend.services.image_price_service.domain.policy import (
    DEFAULT_PRICE_RISK_POLICY,
    PriceRiskPolicy,
)

ConditionReviewer = Callable[
    [dict[str, object]],
    DeepAnalysisReview | dict[str, object] | None,
]
RepriceCallback = Callable[
    [MarketplaceCondition, str],
    tuple[MarketPriceEstimate, ...],
]
PriceDirection = Literal["within", "under", "over"]


class FusionDecisionEngine:
    """以同一份 PriceRiskPolicy 執行價格規則與狀態複核。"""

    def __init__(
        self,
        policy: PriceRiskPolicy = DEFAULT_PRICE_RISK_POLICY,
        *,
        condition_reviewer: ConditionReviewer | None = None,
    ) -> None:
        """注入 domain policy 與選用的狀態複核器。"""
        self.policy = policy
        self._condition_reviewer = condition_reviewer

    @staticmethod
    def _score_from_bands(
        value: float | int,
        bands: tuple[tuple[float | int, int], ...],
    ) -> int:
        """依含等號的上界級距取得分數。"""
        for threshold, score in bands:
            if value <= threshold:
                return score
        return bands[-1][1]

    def _price_components(
        self,
        selling_price: int,
        market_estimate: MarketPriceEstimate,
    ) -> dict[str, float | int | str]:
        """以實際被違反的市場邊界計算相對差距與絕對差額。"""
        if selling_price <= 0:
            raise ValueError("刊登價格必須大於零")
        if not self._is_valid_market_estimate(market_estimate):
            raise ValueError("市場價格區間無效或證據不足")

        if selling_price < market_estimate.low_price:
            direction: PriceDirection = "under"
            boundary = market_estimate.low_price
            absolute_gap = boundary - selling_price
            relative_gap = absolute_gap / boundary
            relative_score = self._score_from_bands(
                relative_gap,
                self.policy.underprice_relative_bands,
            )
            score_cap = self.policy.maximum_score
        elif selling_price > market_estimate.high_price:
            direction = "over"
            boundary = market_estimate.high_price
            absolute_gap = selling_price - boundary
            relative_gap = absolute_gap / boundary
            relative_score = self._score_from_bands(
                relative_gap,
                self.policy.overprice_relative_bands,
            )
            score_cap = self.policy.overprice_score_cap
        else:
            return {
                "direction": "within",
                "boundary": 0,
                "absolute_gap": 0,
                "relative_gap": 0.0,
                "relative_score": 0,
                "absolute_bonus": 0,
                "score": 0.0,
            }

        absolute_bonus = self._score_from_bands(
            absolute_gap,
            self.policy.absolute_gap_bands,
        )
        if market_estimate.reference_mode == "median_low_sample":
            score_cap = min(
                score_cap,
                self.policy.small_sample_score_cap,
            )
        score = min(
            relative_score + absolute_bonus,
            score_cap,
            self.policy.maximum_score,
        )
        return {
            "direction": direction,
            "boundary": boundary,
            "absolute_gap": absolute_gap,
            "relative_gap": relative_gap,
            "relative_score": relative_score,
            "absolute_bonus": absolute_bonus,
            "score": float(score),
        }

    def _calculate_price_score(
        self,
        selling_price: int,
        market_estimate: MarketPriceEstimate,
    ) -> float:
        """依市場區間與 policy 計算純價格風險分數。"""
        return float(
            self._price_components(
                selling_price,
                market_estimate,
            )["score"]
        )

    def _risk_label_from_score(self, score: float) -> RiskLabel:
        """使用 policy 的 LOW／MEDIUM 邊界轉換風險標籤。"""
        if score > self.policy.medium_score_max:
            return "HIGH"
        if score > self.policy.low_score_max:
            return "MEDIUM"
        return "LOW"

    def _has_price_risk(
        self,
        selling_price: int,
        market_estimate: MarketPriceEstimate,
    ) -> bool:
        """判斷結構化價格結果是否已達 MEDIUM／HIGH。"""
        try:
            score = self._calculate_price_score(
                selling_price,
                market_estimate,
            )
        except ValueError:
            return False
        return score > self.policy.low_score_max

    def _enforce_price_risk(
        self,
        decision: dict[str, Any],
        selling_price: int,
        market_estimate: MarketPriceEstimate,
    ) -> dict[str, Any]:
        """只對完整 IQR、高可信且已達 HIGH 的最終規則套用硬性下限。"""
        price_score = self._calculate_price_score(
            selling_price,
            market_estimate,
        )
        if (
            market_estimate.reference_mode != "iqr"
            or market_estimate.confidence
            < self.policy.minimum_market_confidence
            or price_score <= self.policy.medium_score_max
        ):
            return decision

        decision_score = decision.get("risk_score")
        enforced_score = (
            max(float(decision_score), price_score)
            if isinstance(decision_score, (int, float))
            else price_score
        )
        enforced_score = min(
            enforced_score,
            float(self.policy.maximum_score),
        )
        evidence = [str(item) for item in decision.get("evidence") or []]
        price_evidence = "完整 IQR 市場資料的 HIGH 價格規則下限"
        if price_evidence not in evidence:
            evidence.append(price_evidence)
        return {
            **decision,
            "risk_label": self._risk_label_from_score(enforced_score),
            "risk_score": enforced_score,
            "evidence": evidence,
        }

    def _requires_deep_analysis(
        self,
        base_score: float,
        condition: MarketplaceCondition,
        condition_extraction_confidence: float,
        text: str,
        condition_detail: str,
        condition_source_text: str,
        condition_has_conflict: bool,
    ) -> bool:
        """只在價格有風險且有需要複核的有限狀態證據時呼叫 LLM。"""
        has_price_risk = base_score > self.policy.low_score_max
        has_reviewable_evidence = any(
            value.strip()
            for value in (text, condition_detail, condition_source_text)
        )
        condition_needs_review = (
            condition == MarketplaceCondition.UNKNOWN
            or condition_extraction_confidence
            <= self.policy.condition_llm_correction_max_confidence
            or condition_has_conflict
        )
        return (
            has_price_risk
            and has_reviewable_evidence
            and condition_needs_review
        )

    def _alt_deep_result(
        self,
        base_result: dict[str, Any],
    ) -> dict[str, Any]:
        """LLM 無效時原樣保留 MEDIUM／HIGH 價格規則結果。"""
        score = base_result.get("risk_score")
        if not isinstance(score, (int, float)):
            raise ValueError("_alt_deep_result() 需要數字規則分數")
        label = self._risk_label_from_score(float(score))
        if label not in {"MEDIUM", "HIGH"}:
            raise ValueError(
                "_alt_deep_result() 只接受 MEDIUM／HIGH 規則結果"
            )

        evidence = [str(item) for item in base_result.get("evidence") or []]
        fallback_evidence = "LLM 狀態複核不可用，保留原始價格規則結果"
        if fallback_evidence not in evidence:
            evidence.append(fallback_evidence)
        reason = str(base_result.get("reason") or "價格規則評估完成")
        return {
            **base_result,
            "risk_label": label,
            "risk_score": float(score),
            "reason": f"{reason}；未套用任何狀態修正",
            "evidence": evidence,
            "confidence": float(base_result.get("confidence") or 0.0),
            "decision_layer": "llm_simulated",
        }

    def _call_llm_deep_analysis(
        self,
        product_context: dict[str, object],
    ) -> DeepAnalysisReview | None:
        """呼叫注入的狀態複核器並以 DeepAnalysisReview 驗證輸出。"""
        if self._condition_reviewer is None:
            return None
        try:
            payload = self._condition_reviewer(product_context)
            if payload is None:
                return None
            if isinstance(payload, DeepAnalysisReview):
                return payload
            return DeepAnalysisReview.model_validate(payload)
        except (ValidationError, TypeError, ValueError):
            return None
        except Exception:
            return None

    def _is_valid_market_estimate(
        self,
        estimate: MarketPriceEstimate,
    ) -> bool:
        """驗證決策使用的市場區間、資料量及可信度。"""
        maximum = self.policy.maximum_supported_price
        return (
            estimate.status == "success"
            and estimate.source != "not_evaluated"
            and estimate.sample_count >= self.policy.minimum_market_samples
            and estimate.site_count >= self.policy.minimum_market_sites
            and estimate.confidence >= self.policy.minimum_market_confidence
            and 0 < estimate.low_price <= maximum
            and 0 < estimate.median_price <= maximum
            and 0 < estimate.high_price <= maximum
            and estimate.low_price
            <= estimate.median_price
            <= estimate.high_price
        )

    def _rule_result_for_estimate(
        self,
        selling_price: int,
        estimate: MarketPriceEstimate,
    ) -> dict[str, Any]:
        components = self._price_components(selling_price, estimate)
        score = float(components["score"])
        direction = str(components["direction"])
        if direction == "within":
            reason = (
                f"刊登價格 {selling_price} 位於"
                f" {estimate.low_price}～{estimate.high_price} 市場區間，"
                "相對差距 0.00%、絕對差額 0"
            )
            price_evidence = (
                "刊登價格位於有效市場區間：相對分數 0、"
                "絕對差額加分 0"
            )
        else:
            direction_text = "低於" if direction == "under" else "高於"
            reason = (
                f"刊登價格 {selling_price} {direction_text}市場邊界"
                f" {components['boundary']}，相對差距"
                f" {float(components['relative_gap']):.2%}、絕對差額"
                f" {components['absolute_gap']}"
            )
            price_evidence = (
                f"{direction_text}行情：相對分數"
                f" {components['relative_score']}、絕對差額加分"
                f" {components['absolute_bonus']}"
            )

        return {
            "risk_label": self._risk_label_from_score(score),
            "risk_score": score,
            "reason": reason,
            "evidence": [
                (
                    f"{estimate.condition.value} 市場區間"
                    f" {estimate.low_price}～{estimate.high_price}"
                ),
                price_evidence,
            ],
            "confidence": estimate.confidence,
            "market_price_source": estimate.source,
            "reference_mode": estimate.reference_mode,
            "condition": estimate.condition,
        }

    def _evaluate_market_paths(
        self,
        *,
        selling_price: int,
        condition: MarketplaceCondition,
        market_estimates: tuple[MarketPriceEstimate, ...],
    ) -> dict[str, Any]:
        if condition != MarketplaceCondition.UNKNOWN:
            if len(market_estimates) != 1:
                return self._decision_error(
                    "MARKET_PRICE_INSUFFICIENT",
                    "已知商品狀態必須且只能取得一個相同狀態的有效市場結果",
                    condition=condition,
                )
            estimate = market_estimates[0]
            if (
                estimate.condition != condition
                or not self._is_valid_market_estimate(estimate)
            ):
                return self._market_estimate_error(
                    market_estimates,
                    condition,
                )
            result = self._rule_result_for_estimate(
                selling_price,
                estimate,
            )
            return result

        if len(market_estimates) != 2:
            return self._decision_error(
                "MARKET_PRICE_DUAL_PATH_INSUFFICIENT",
                "商品狀態未知時必須同時取得全新與二手市場結果",
                condition=condition,
            )
        by_condition = {
            estimate.condition: estimate
            for estimate in market_estimates
            if self._is_valid_market_estimate(estimate)
        }
        if set(by_condition) != {
            MarketplaceCondition.NEW,
            MarketplaceCondition.USED,
        }:
            return self._market_estimate_error(
                market_estimates,
                condition,
            )

        new_result = self._rule_result_for_estimate(
            selling_price,
            by_condition[MarketplaceCondition.NEW],
        )
        used_result = self._rule_result_for_estimate(
            selling_price,
            by_condition[MarketplaceCondition.USED],
        )
        path_results = (new_result, used_result)
        labels = {
            str(result["risk_label"])
            for result in path_results
        }
        if "LOW" in labels:
            merged_label: RiskLabel = "LOW"
        elif labels == {"HIGH"}:
            merged_label = "HIGH"
        else:
            merged_label = "MEDIUM"

        matching_results = [
            result
            for result in path_results
            if result["risk_label"] == merged_label
        ]
        merged_score = min(
            float(result["risk_score"])
            for result in matching_results
        )
        sources = {
            result["market_price_source"]
            for result in path_results
        }
        market_source: MarketPriceSource = (
            next(iter(sources))
            if len(sources) == 1
            else "not_evaluated"
        )
        return {
            "risk_label": merged_label,
            "risk_score": merged_score,
            "reason": (
                "商品狀態未知，已分別比較全新與二手市場區間，"
                f"依保守規則採用 {merged_label}；"
                f"全新：{new_result['reason']}；"
                f"二手：{used_result['reason']}"
            ),
            "evidence": [
                *[
                    str(item)
                    for result in path_results
                    for item in result["evidence"]
                ],
            ],
            "confidence": min(
                float(result["confidence"])
                for result in path_results
            ),
            "market_price_source": market_source,
            "reference_mode": (
                "iqr"
                if all(
                    result["reference_mode"] == "iqr"
                    for result in path_results
                )
                else "median_low_sample"
            ),
            "condition": MarketplaceCondition.UNKNOWN,
        }

    def _market_estimate_error(
        self,
        estimates: tuple[MarketPriceEstimate, ...],
        condition: MarketplaceCondition,
    ) -> dict[str, Any]:
        has_candidate_data = any(
            estimate.status == "insufficient"
            or estimate.sample_count > 0
            for estimate in estimates
        )
        return self._decision_error(
            (
                "MARKET_PRICE_INSUFFICIENT"
                if has_candidate_data
                else "MARKET_PRICE_NOT_FOUND"
            ),
            "市場價格資料不足或區間無效",
            condition=condition,
        )

    @staticmethod
    def _decision_error(
        error_code: str,
        reason: str,
        *,
        condition: MarketplaceCondition = MarketplaceCondition.UNKNOWN,
    ) -> dict[str, Any]:
        return {
            "risk_label": "UNKNOWN",
            "risk_score": "未知",
            "reason": reason,
            "evidence": [],
            "confidence": 0.0,
            "decision_layer": "decision_error",
            "market_price_source": "not_evaluated",
            "condition": condition,
            "error_code": error_code,
        }

    @staticmethod
    def _normalize_review_text(value: str) -> str:
        return re.sub(r"[\s・·:：_\-]+", "", value).casefold()

    def _review_is_acceptable(
        self,
        review: DeepAnalysisReview,
        text: str,
        condition_detail: str,
        condition_source_text: str,
    ) -> bool:
        evidence = self._normalize_review_text(review.condition_evidence)
        sources = tuple(
            self._normalize_review_text(value)
            for value in (text, condition_detail, condition_source_text)
            if value.strip()
        )
        return (
            review.review_confidence
            >= self.policy.llm_review_min_confidence
            and bool(evidence)
            and any(evidence in source for source in sources)
        )

    def _apply_review(
        self,
        *,
        base_result: dict[str, Any],
        review: DeepAnalysisReview,
        selling_price: int,
        original_condition: MarketplaceCondition,
        original_condition_detail: str,
        original_market_estimates: tuple[MarketPriceEstimate, ...],
        condition_extraction_confidence: float,
        reprice: RepriceCallback | None,
    ) -> dict[str, Any]:
        same_condition = review.reviewed_condition == original_condition
        same_detail = (
            self._normalize_review_text(review.condition_detail)
            == self._normalize_review_text(original_condition_detail)
        )
        if same_condition and same_detail:
            evidence = [
                *[str(item) for item in base_result.get("evidence") or []],
                review.condition_evidence,
            ]
            confirmed_result = {
                **base_result,
                "reason": (
                    f"{base_result['reason']}；LLM 已確認原始商品狀態"
                ),
                "evidence": evidence,
                "confidence": min(
                    float(base_result.get("confidence") or 0.0),
                    review.review_confidence,
                ),
                "decision_layer": "llm",
                "condition_detail": original_condition_detail,
            }
            return self._enforce_final_decision(
                confirmed_result,
                selling_price=selling_price,
                condition=original_condition,
                market_estimates=original_market_estimates,
            )

        if (
            condition_extraction_confidence
            > self.policy.condition_llm_correction_max_confidence
            or reprice is None
        ):
            return self._enforce_final_decision(
                self._alt_deep_result(base_result),
                selling_price=selling_price,
                condition=original_condition,
                market_estimates=original_market_estimates,
            )

        try:
            repriced_estimates = tuple(
                reprice(
                    review.reviewed_condition,
                    review.condition_detail,
                )
            )
        except Exception:
            return self._decision_error(
                "MARKET_PRICE_REPRICE_FAILED",
                "接受狀態修正後重新查價失敗",
                condition=review.reviewed_condition,
            )

        repriced_result = self._evaluate_market_paths(
            selling_price=selling_price,
            condition=review.reviewed_condition,
            market_estimates=repriced_estimates,
        )
        if repriced_result.get("decision_layer") == "decision_error":
            return repriced_result

        evidence = [
            *[str(item) for item in repriced_result.get("evidence") or []],
            review.condition_evidence,
        ]
        reviewed_result = {
            **repriced_result,
            "reason": (
                f"{repriced_result['reason']}；"
                f"LLM 狀態複核：{review.reason}"
            ),
            "evidence": evidence,
            "confidence": min(
                float(repriced_result.get("confidence") or 0.0),
                review.review_confidence,
            ),
            "decision_layer": "llm",
            "condition": review.reviewed_condition,
            "condition_detail": review.condition_detail,
        }
        return self._enforce_final_decision(
            reviewed_result,
            selling_price=selling_price,
            condition=review.reviewed_condition,
            market_estimates=repriced_estimates,
        )

    def _enforce_final_decision(
        self,
        decision: dict[str, Any],
        *,
        selling_price: int,
        condition: MarketplaceCondition,
        market_estimates: tuple[MarketPriceEstimate, ...],
    ) -> dict[str, Any]:
        """在狀態複核完成後，才對最終單一路徑套用價格硬性下限。"""
        if (
            condition == MarketplaceCondition.UNKNOWN
            or len(market_estimates) != 1
            or not self._is_valid_market_estimate(market_estimates[0])
        ):
            return decision
        return self._enforce_price_risk(
            decision,
            selling_price,
            market_estimates[0],
        )

    def evaluate(
        self,
        *,
        product_name: str,
        selling_price: int,
        market_estimates: tuple[MarketPriceEstimate, ...] = (),
        condition: MarketplaceCondition = MarketplaceCondition.UNKNOWN,
        condition_detail: str = "",
        condition_source_text: str = "",
        condition_extraction_confidence: float = 0.0,
        condition_has_conflict: bool = False,
        reprice: RepriceCallback | None = None,
        brand_model: str = "",
        text: str = "",
        market_price: int | None = None,
        market_price_source: MarketPriceSource = "not_evaluated",
    ) -> dict[str, Any]:
        """執行結構化價格規則，並在符合條件時選用 LLM 複核狀態。"""
        del brand_model, market_price, market_price_source

        if not product_name.strip():
            return self._decision_error(
                "PRODUCT_NAME_MISSING",
                "缺少商品標題，無法執行價格判斷",
                condition=condition,
            )
        if selling_price <= 0:
            return self._decision_error(
                "LISTED_PRICE_INVALID",
                "刊登價格必須大於零",
                condition=condition,
            )
        if selling_price > self.policy.maximum_supported_price:
            return self._decision_error(
                "PRICE_OUT_OF_SUPPORTED_RANGE",
                "刊登價格超出服務支援範圍",
                condition=condition,
            )

        base_result = self._evaluate_market_paths(
            selling_price=selling_price,
            condition=condition,
            market_estimates=tuple(market_estimates),
        )
        if base_result.get("decision_layer") == "decision_error":
            return base_result

        base_score = float(base_result["risk_score"])
        if not self._requires_deep_analysis(
            base_score,
            condition,
            condition_extraction_confidence,
            text,
            condition_detail,
            condition_source_text,
            condition_has_conflict,
        ):
            fast_result = {
                **base_result,
                "decision_layer": "fast",
                "condition_detail": condition_detail,
            }
            return self._enforce_final_decision(
                fast_result,
                selling_price=selling_price,
                condition=condition,
                market_estimates=market_estimates,
            )

        product_context: dict[str, object] = {
            "product_name": " ".join(product_name.split()),
            "text": " ".join(text.split()),
            "condition": condition.value,
            "condition_detail": condition_detail,
            "condition_source_text": condition_source_text,
            "condition_extraction_confidence": (
                condition_extraction_confidence
            ),
        }
        review_payload = self._call_llm_deep_analysis(product_context)
        try:
            review = (
                review_payload
                if isinstance(review_payload, DeepAnalysisReview)
                else DeepAnalysisReview.model_validate(review_payload)
            )
        except (ValidationError, TypeError, ValueError):
            review = None

        if (
            review is None
            or not self._review_is_acceptable(
                review,
                text,
                condition_detail,
                condition_source_text,
            )
        ):
            return self._enforce_final_decision(
                self._alt_deep_result(base_result),
                selling_price=selling_price,
                condition=condition,
                market_estimates=market_estimates,
            )

        return self._apply_review(
            base_result=base_result,
            review=review,
            selling_price=selling_price,
            original_condition=condition,
            original_condition_detail=condition_detail,
            original_market_estimates=market_estimates,
            condition_extraction_confidence=(
                condition_extraction_confidence
            ),
            reprice=reprice,
        )


fusion_decision_engine = FusionDecisionEngine()
