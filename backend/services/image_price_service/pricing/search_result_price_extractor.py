"""使用一次結構化 LLM 呼叫整理搜尋結果中的同商品價格。"""

import json
import logging
import re
import unicodedata
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from backend.config import settings
from backend.services.dto.price_analysis import MarketPriceCandidateEvidence
from backend.services.image_price_service.domain.models import MarketplaceCondition
from backend.services.image_price_service.domain.policy import (
    DEFAULT_PRICE_RISK_POLICY,
    PriceRiskPolicy,
)
from backend.services.image_price_service.pricing.url_utils import (
    normalize_url,
)

logger = logging.getLogger(__name__)

_ZERO_WIDTH_CHARACTERS = frozenset(
    {
        "\u200b",
        "\u200c",
        "\u200d",
        "\ufeff",
    }
)
_ALLOWED_SYMBOLS = frozenset({"$", "+", "×", "=", "~"})

_SYSTEM_PROMPT = """
你是台灣購物搜尋結果的商品價格整理器。請一次分析所有輸入結果，擷取其中明確的
商品售價並判斷品況，最後只輸出符合下列格式的有效 json 物件。

規則：
1. 只負責從每筆 title 與 snippet 擷取能明確配對到商品的售價。
2. title 或 snippet 可能以分號列出多項商品，價格也可能出現在商品描述之前或之後。
   必須依語意將每個價格配對到正確商品，不得假設價格位於固定位置。
3. 只接受明確的一次性商品售價；排除運費、折扣金額、定金、月付、分期、面交地點、
   年份、尺寸、比例、商品編號、價格範圍，以及無法確定是售價的數字。
4. condition 只能是 new、used 或 unknown。全新未拆為 new；二手、拆封、拆擺、
   中古、展示品為 used；沒有足夠文字判斷時為 unknown。
5. result_index 必須引用輸入中的索引。evidence 必須逐字取自該筆 title 或 snippet，
   並包含足以核對該價格的文字。不得捏造商品、價格、品況或來源。
6. 同一筆結果若有多個可明確配對的獨立商品售價，可輸出多筆；若是合售，
   只輸出合售總價，不可自行拆算單價。沒有可靠候選時回傳空陣列。

輸出格式：
{
  "candidates": [
    {
      "result_index": 0,
      "price": 840,
      "condition": "used",
      "evidence": "二手拆擺七支合售. $840"
    }
  ]
}

欄位規則：
- candidates 是價格候選陣列。
- result_index 必須是對應輸入 results 的非負整數索引。
- price 必須是大於 0 的新台幣整數，不得包含貨幣符號或千分位逗號。
- condition 只能是 "new"、"used" 或 "unknown"。
- evidence 必須逐字取自對應結果的 title 或 snippet。
- 沒有可靠候選時輸出 {"candidates": []}。
- 不得輸出 candidates 以外的欄位或額外說明。
""".strip()


class ExtractedSearchPrice(BaseModel):
    """LLM 從單筆搜尋結果辨識出的價格候選。"""

    model_config = ConfigDict(extra="forbid")

    result_index: int = Field(ge=0)
    price: int = Field(gt=0)
    condition: MarketplaceCondition
    evidence: str = Field(min_length=1)


class SearchPriceExtraction(BaseModel):
    """一次搜尋結果整理的結構化輸出。"""

    model_config = ConfigDict(extra="forbid")

    candidates: list[ExtractedSearchPrice] = Field(default_factory=list)


class GroqSearchResultPriceExtractor:
    """延遲建立 Groq client，並以一次 structured-output 呼叫整理價格。"""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        structured_llm: Any | None = None,
    ) -> None:
        self._api_key = api_key.strip()
        self._model_name = model_name.strip()
        self._structured_llm = structured_llm

    def _get_structured_llm(self) -> Any:
        if self._structured_llm is not None:
            return self._structured_llm
        if not self._api_key:
            raise RuntimeError("尚未設定 GROQ_API_KEY，無法整理搜尋結果價格")
        if not self._model_name:
            raise RuntimeError("尚未設定價格整理模型")

        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=self._model_name,
            temperature=0,
            reasoning_effort="none",
            api_key=SecretStr(self._api_key),
        )
        self._structured_llm = llm.with_structured_output(
            SearchPriceExtraction,
            method="json_mode",
        )
        return self._structured_llm

    def extract(
        self,
        search_results: list[dict[str, Any]],
        condition: MarketplaceCondition,
        *,
        product_query: str,
        policy: PriceRiskPolicy = DEFAULT_PRICE_RISK_POLICY,
    ) -> list[MarketPriceCandidateEvidence]:
        """將一批搜尋結果交給 LLM，並驗證其結構化候選。"""
        normalized_results = _normalize_search_results(search_results)
        if not normalized_results or not product_query.strip():
            return []

        payload = {
            "results": [
                {
                    "result_index": index,
                    "title": result["title"],
                    "snippet": result["snippet"],
                }
                for index, result in enumerate(normalized_results)
            ],
        }
        response = self._get_structured_llm().invoke(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", f"輸入：{json.dumps(payload, ensure_ascii=False)}"),
            ]
        )
        extraction = (
            response
            if isinstance(response, SearchPriceExtraction)
            else SearchPriceExtraction.model_validate(response)
        )
        return _validate_candidates(
            extraction,
            normalized_results,
            condition,
            policy,
        )


def _normalize_search_results(
    search_results: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """只保留 LLM 與來源回填所需的可信欄位。"""
    normalized: list[dict[str, str]] = []
    for result in search_results:
        if not isinstance(result, dict):
            continue
        title = str(result.get("title", "")).strip()
        snippet = str(result.get("snippet", "")).strip()
        link = str(result.get("link", "")).strip()

        title = _sanitize_llm_text(title)
        snippet = _sanitize_llm_text(snippet)
        normalized_url = normalize_url(link)
        if not normalized_url or not (title or snippet):
            continue
        normalized.append(
            {
                "title": title,
                "snippet": snippet,
                "link": link,
                "normalized_url": normalized_url,
            }
        )
    return normalized


def _sanitize_llm_text(text: str) -> str:
    """清除 output 的隱形與裝飾字元。"""
    cleaned: list[str] = []
    for character in unicodedata.normalize("NFKC", text):
        if character in _ZERO_WIDTH_CHARACTERS:
            continue
        if character.isspace():
            cleaned.append(" ")
            continue

        category = unicodedata.category(character)
        family = category[0]
        if (
            family in {"L", "N", "P"}
            or category == "Sc"
            or character in _ALLOWED_SYMBOLS
        ):
            cleaned.append(character)
            continue
        if (
            family == "M"
            and cleaned
            and unicodedata.category(cleaned[-1]).startswith("L")
        ):
            cleaned.append(character)
            continue

        # 不在白名單的 emoji、圖形與控制字元以空白隔開，
        # 避免移除後將原本分離的兩個詞黏在一起。
        cleaned.append(" ")

    return re.sub(r"\s+", " ", "".join(cleaned)).strip()


def _validate_candidates(
    extraction: SearchPriceExtraction,
    search_results: list[dict[str, str]],
    condition: MarketplaceCondition,
    policy: PriceRiskPolicy,
) -> list[MarketPriceCandidateEvidence]:
    """以原始索引回填來源，並執行非語意性的安全驗證。"""
    candidates: list[MarketPriceCandidateEvidence] = []
    for candidate_index, extracted in enumerate(extraction.candidates):
        if extracted.result_index >= len(search_results):
            continue
        if (
            condition is not MarketplaceCondition.UNKNOWN
            and extracted.condition is not condition
        ):
            continue
        if extracted.condition is MarketplaceCondition.UNKNOWN:
            continue
        if extracted.price > policy.maximum_supported_price:
            continue

        source = search_results[extracted.result_index]
        source_text = " ".join(
            part
            for part in (source["title"], source["snippet"])
            if part
        )
        evidence = extracted.evidence.strip()
        if evidence not in source_text:
            continue

        candidates.append(
            MarketPriceCandidateEvidence(
                candidate_id=(
                    f"{source['normalized_url']}#{extracted.price}"
                    f"#{extracted.condition.value}#{candidate_index}"
                ),
                title=source["title"],
                price=extracted.price,
                condition=extracted.condition,
                url=source["link"],
                evidence=evidence,
            )
        )
    return candidates


default_search_result_price_extractor = GroqSearchResultPriceExtractor(
    api_key=settings.GROQ_API_KEY.get_secret_value(),
    model_name=settings.PRODUCT_MODEL_NAME,
)


def extract_prices_from_search_results(
    search_results: list[dict[str, Any]],
    condition: MarketplaceCondition,
    *,
    product_query: str,
    policy: PriceRiskPolicy = DEFAULT_PRICE_RISK_POLICY,
    extractor: GroqSearchResultPriceExtractor | None = None,
) -> list[MarketPriceCandidateEvidence]:
    """相容函式：以一次 LLM 呼叫整理整批搜尋結果。"""
    active_extractor = extractor or default_search_result_price_extractor
    return active_extractor.extract(
        search_results,
        condition,
        product_query=product_query,
        policy=policy,
    )


__all__ = [
    "ExtractedSearchPrice",
    "GroqSearchResultPriceExtractor",
    "SearchPriceExtraction",
    "extract_prices_from_search_results",
]
