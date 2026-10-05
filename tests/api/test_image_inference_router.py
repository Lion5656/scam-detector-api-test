import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.api.routers import image_price_validation
from backend.services.dto.price_analysis import (ImagePriceAnalysisResult,
                                                 MarketPriceEstimate)
from backend.services.image_price_service.domain.models import \
    MarketplaceCondition

client_app = FastAPI()
client_app.include_router(image_price_validation.router)
client = TestClient(client_app)


def _analysis_result(**overrides) -> ImagePriceAnalysisResult:
    values = {
        "filename": "ad.png",
        "content_type": "image/png",
        "extracted_text": "iphone 15 特價 10000 元",
        "product_name": "Apple iPhone 15",
        "brand_model": "Apple iPhone 15",
        "listed_price": 10000,
        "market_price": 27900,
        "market_price_source": "online",
        "market_price_estimates": (
            MarketPriceEstimate(
                status="success",
                condition=MarketplaceCondition.USED,
                reference_mode="iqr",
                median_price=27_900,
                low_price=25_000,
                high_price=30_000,
                sample_count=5,
                site_count=3,
                source="online",
                confidence=0.8,
            ),
        ),
        "search_tools": ["serp_api"],
        "risk_label": "HIGH",
        "score": 90.0,
        "reason": "低於行情",
        "evidence": ["低於行情 50% 規則觸發"],
        "confidence": 0.8,
        "decision_layer": "llm_simulated",
        "seller_name": "Wei-Cheng Fang",
        "condition": MarketplaceCondition.USED,
        "condition_detail": "近全新",
        "condition_source_text": "狀況 二手・近全新",
        "condition_extraction_confidence": 0.97,
        "extraction_confidence": 0.91,
        "price_source_text": "NT$13,000",
        "extraction_warnings": [],
    }
    values.update(overrides)
    return ImagePriceAnalysisResult(**values)


def test_image_price_analysis_result_rejects_stale_field_names():
    with pytest.raises(ValidationError):
        _analysis_result(selling_price=13000)

    with pytest.raises(ValidationError):
        _analysis_result(price_extraction_confidence=0.9)


def test_image_price_analysis_result_has_only_current_fields():
    assert set(ImagePriceAnalysisResult.model_fields) == {
        "filename",
        "content_type",
        "extracted_text",
        "success",
        "error_code",
        "message",
        "product_name",
        "brand_model",
        "listed_price",
        "market_price",
        "market_price_source",
        "market_price_estimates",
        "risk_label",
        "score",
        "reason",
        "evidence",
        "confidence",
        "decision_layer",
        "search_tools",
        "marketplace_layout",
        "marketplace_confidence",
        "extraction_confidence",
        "price_source_text",
        "price_extraction_reason",
        "seller_name",
        "condition",
        "condition_detail",
        "condition_source_text",
        "condition_extraction_confidence",
        "extraction_warnings",
    }


def test_analyze_image_price_maps_service_result_to_response(monkeypatch):
    called = {}

    def fake_detector(data: bytes, filename: str, content_type: str):
        called.update(data=data, filename=filename, content_type=content_type)
        return _analysis_result(filename=filename, content_type=content_type)

    monkeypatch.setattr(
        image_price_validation.image_price_analyzer,
        "image_price_detector",
        fake_detector,
    )

    response = client.post(
        "/v1/analyze/price",
        files={"file": ("ad.png", b"fake-bytes", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "product_name",
        "condition",
        "condition_detail",
        "listed_price",
        "market_price",
        "seller_name",
        "risk_label",
        "risk_score",
        "decision_layer",
        "error_code",
        "result",
        "extraction_confidence",
        "debug",
    }
    assert payload["product_name"] == "Apple iPhone 15"
    assert payload["listed_price"] == 10000
    assert payload["market_price"] == 27900
    assert payload["seller_name"] == "Wei-Cheng Fang"
    assert payload["risk_label"] == "HIGH"
    assert payload["risk_score"] == 90.0
    assert payload["decision_layer"] == "llm_simulated"
    assert payload["condition"] == "used"
    assert payload["condition_detail"] == "近全新"
    assert payload["result"] == "低於行情"
    assert payload["debug"] == {
        "search_tools": ["serp_api"],
        "market_price_source": "online",
        "market_price_estimates": [
            {
                "status": "success",
                "condition": "used",
                "reference_mode": "iqr",
                "median_price": 27900,
                "low_price": 25000,
                "high_price": 30000,
                "sample_count": 5,
                "site_count": 3,
                "source": "online",
                "confidence": 0.8,
            }
        ],
        "condition_source_text": "狀況 二手・近全新",
        "condition_extraction_confidence": 0.97,
        "price_source_text": "NT$13,000",
        "warnings": [],
    }
    assert called == {
        "data": b"fake-bytes",
        "filename": "ad.png",
        "content_type": "image/png",
    }


def test_analyze_image_price_returns_clear_invalid_source_payload(monkeypatch):
    monkeypatch.setattr(
        image_price_validation.image_price_analyzer,
        "image_price_detector",
        lambda **kwargs: _analysis_result(
            success=False,
            error_code="INVALID_IMAGE_SOURCE",
            message="圖片格式錯誤，來源需為 FB Marketplace 商品頁截圖",
            reason="圖片格式錯誤，來源需為 FB Marketplace 商品頁截圖",
            listed_price=None,
            market_price=0,
            market_price_source="not_evaluated",
            search_tools=[],
            risk_label="UNKNOWN",
            score="未知",
            decision_layer="decision_error",
        ),
    )

    response = client.post(
        "/v1/analyze/price",
        files={"file": ("not-marketplace.png", b"fake-bytes", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["product_name"] is None
    assert payload["listed_price"] is None
    assert payload["market_price"] is None
    assert payload["seller_name"] is None
    assert payload["risk_label"] == "UNKNOWN"
    assert payload["decision_layer"] == "decision_error"
    assert payload["error_code"] == "INVALID_IMAGE_SOURCE"
    assert payload["result"] == "圖片格式錯誤，來源需為 FB Marketplace 商品頁截圖"
    assert payload["debug"]["market_price_source"] == "not_evaluated"
    assert payload["debug"]["search_tools"] == []
    assert payload["debug"]["warnings"] == ["圖片格式錯誤，來源需為 FB Marketplace 商品頁截圖"]


def test_analyze_image_price_exposes_search_tools(monkeypatch):
    monkeypatch.setattr(
        image_price_validation.image_price_analyzer,
        "image_price_detector",
        lambda **kwargs: _analysis_result(search_tools=["serp_api", "ddgs"]),
    )

    response = client.post(
        "/v1/analyze/price",
        files={"file": ("ad.png", b"fake-bytes", "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["debug"]["search_tools"] == ["serp_api", "ddgs"]


@pytest.mark.parametrize("decision_layer", ["fast", "llm", "llm_simulated"])
def test_analyze_image_price_preserves_success_decision_layers(
    monkeypatch,
    decision_layer,
):
    monkeypatch.setattr(
        image_price_validation.image_price_analyzer,
        "image_price_detector",
        lambda **kwargs: _analysis_result(decision_layer=decision_layer),
    )

    response = client.post(
        "/v1/analyze/price",
        files={"file": ("ad.png", b"fake-bytes", "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["decision_layer"] == decision_layer
    assert response.json()["decision_layer"] != "source_validation"


def test_analyze_image_price_invalid_content_type():
    response = client.post(
        "/v1/analyze/price",
        files={"file": ("ad.txt", b"text-bytes", "text/plain")},
    )

    assert response.status_code == 400
    assert "僅支援" in response.json()["detail"]


def test_analyze_image_price_rejects_empty_file():
    response = client.post(
        "/v1/analyze/price",
        files={"file": ("ad.png", b"", "image/png")},
    )

    assert response.status_code == 400
    assert "不可為空" in response.json()["detail"]


def test_image_price_router_has_no_text_analyzer_dependency():
    assert not hasattr(image_price_validation, "text_analyzer")
