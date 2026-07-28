"""集中定義商品價格風險使用的所有數字門檻。"""

from math import isinf, isnan

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PriceRiskPolicy(BaseModel):
    """不可變且不依賴環境設定的價格風險數字門檻。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    underprice_relative_bands: tuple[tuple[float, int], ...]
    overprice_relative_bands: tuple[tuple[float, int], ...]
    absolute_gap_bands: tuple[tuple[int, int], ...]
    maximum_supported_price: int = Field(gt=0)
    low_score_max: int = Field(ge=0, le=100)
    medium_score_max: int = Field(ge=0, le=100)
    maximum_score: int = Field(ge=0, le=100)
    overprice_score_cap: int = Field(ge=0, le=100)
    minimum_market_confidence: float = Field(ge=0.0, le=1.0)
    minimum_market_samples: int = Field(ge=3)
    minimum_market_sites: int = Field(ge=1)
    minimum_iqr_samples: int = Field(ge=5)
    small_sample_relative_tolerance: float = Field(gt=0.0, lt=1.0)
    small_sample_score_cap: int = Field(ge=0, le=100)
    llm_review_min_confidence: float = Field(ge=0.0, le=1.0)
    condition_llm_correction_max_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    @staticmethod
    def _validate_relative_bands(
        bands: tuple[tuple[float, int], ...],
        band_name: str,
        maximum_score: int,
    ) -> None:
        if not bands:
            raise ValueError(f"{band_name}不得為空")

        previous_threshold = 0.0
        for index, (threshold, score) in enumerate(bands):
            if isnan(threshold) or threshold <= previous_threshold:
                raise ValueError(f"{band_name}門檻必須嚴格遞增且大於零")
            if isinf(threshold) and index != len(bands) - 1:
                raise ValueError(f"{band_name}只有最後一個門檻可以是無限大")
            if not 0 <= score <= maximum_score:
                raise ValueError(
                    f"{band_name}分數必須介於 0 與 maximum_score 之間"
                )
            previous_threshold = threshold

    @staticmethod
    def _validate_absolute_gap_bands(
        bands: tuple[tuple[int, int], ...],
        maximum_supported_price: int,
        maximum_score: int,
    ) -> None:
        if not bands:
            raise ValueError("絕對差額級距不得為空")

        previous_threshold = -1
        for threshold, score in bands:
            if threshold <= previous_threshold:
                raise ValueError("絕對差額級距門檻必須嚴格遞增")
            if not 0 <= score <= maximum_score:
                raise ValueError(
                    "絕對差額級距分數必須介於 0 與 maximum_score 之間"
                )
            previous_threshold = threshold

        if bands[-1][0] != maximum_supported_price:
            raise ValueError(
                "絕對差額最後一個門檻必須等於 maximum_supported_price"
            )

    @model_validator(mode="after")
    def validate_policy(self) -> "PriceRiskPolicy":
        """驗證所有跨欄位門檻關係。"""
        if not (
            self.low_score_max
            < self.medium_score_max
            < self.maximum_score
        ):
            raise ValueError(
                "風險分數邊界必須符合 LOW 小於 MEDIUM 小於 maximum"
            )

        if self.overprice_score_cap > self.medium_score_max:
            raise ValueError("高於行情的分數上限不得超過 MEDIUM")

        if self.minimum_iqr_samples < self.minimum_market_samples:
            raise ValueError("IQR 最低樣本數不得少於市場最低樣本數")

        if self.small_sample_score_cap > self.medium_score_max:
            raise ValueError("小樣本分數上限不得超過 MEDIUM")

        self._validate_relative_bands(
            self.underprice_relative_bands,
            "低於行情相對差距級距",
            self.maximum_score,
        )
        self._validate_relative_bands(
            self.overprice_relative_bands,
            "高於行情相對差距級距",
            self.maximum_score,
        )
        self._validate_absolute_gap_bands(
            self.absolute_gap_bands,
            self.maximum_supported_price,
            self.maximum_score,
        )
        return self


DEFAULT_PRICE_RISK_POLICY = PriceRiskPolicy(
    underprice_relative_bands=(
        (0.10, 10),
        (0.20, 25),
        (0.35, 45),
        (0.50, 65),
        (float("inf"), 85),
    ),
    overprice_relative_bands=(
        (0.15, 10),
        (0.30, 20),
        (0.60, 40),
        (1.00, 55),
        (float("inf"), 70),
    ),
    absolute_gap_bands=(
        (499, 0),
        (1_999, 5),
        (4_999, 10),
        (9_999, 20),
        (100_000, 30),
    ),
    maximum_supported_price=100_000,
    low_score_max=39,
    medium_score_max=79,
    maximum_score=100,
    overprice_score_cap=79,
    minimum_market_confidence=0.60,
    minimum_market_samples=3,
    minimum_market_sites=3,
    minimum_iqr_samples=5,
    small_sample_relative_tolerance=0.25,
    small_sample_score_cap=79,
    llm_review_min_confidence=0.80,
    condition_llm_correction_max_confidence=0.80,
)

__all__ = [
    "DEFAULT_PRICE_RISK_POLICY",
    "PriceRiskPolicy",
]
