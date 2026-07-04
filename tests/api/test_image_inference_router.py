from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers import image_inference
from backend.services.image_service.image_analyzer import ImageTextInsight


client_app = FastAPI()
client_app.include_router(image_inference.router)
client = TestClient(client_app)


def test_analyze_image_scam_low_price_forced_high_risk(monkeypatch):
    def fake_analyze_image_bytes(_: bytes) -> ImageTextInsight:
        return ImageTextInsight(
            extracted_text="iPhone 15 特價 10000 元",
            product_name="Apple iPhone 15",
            brand_model="Apple iPhone 15",
            selling_price=10000,
            market_price=27900,
            market_price_source="online",
            is_high_risk_below_market=True,
        )

    async def fake_hybrid_detector(_: str):
        return {
            "label": "低風險",
            "score": 20.0,
            "reason": "原始文字判定低風險",
            "cls_model_confidence": 0.88,
        }

    monkeypatch.setattr(image_inference.image_analyzer, "analyze_image_bytes", fake_analyze_image_bytes)
    monkeypatch.setattr(image_inference.text_analyzer, "hybrid_detector", fake_hybrid_detector)

    response = client.post(
        "/v1/analyze/image/scam",
        files={"file": ("ad.png", b"fake-bytes", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "高風險"
    assert payload["market_price_source"] == "online"
    assert payload["is_high_risk_below_market"] is True
    assert payload["score"] >= 90
    assert "低於正常市價" in payload["reason"]


def test_analyze_image_scam_half_price_boundary_not_forced(monkeypatch):
    def fake_analyze_image_bytes(_: bytes) -> ImageTextInsight:
        return ImageTextInsight(
            extracted_text="PS5 slim 售價 8790 元",
            product_name="Sony PlayStation 5 Slim",
            brand_model="Sony PlayStation 5 Slim",
            selling_price=8790,
            market_price=17580,
            market_price_source="fallback_local",
            is_high_risk_below_market=False,
        )

    async def fake_hybrid_detector(_: str):
        return {
            "label": "中等風險",
            "score": 55.0,
            "reason": "文字判定為中等風險",
            "cls_model_confidence": 0.62,
        }

    monkeypatch.setattr(image_inference.image_analyzer, "analyze_image_bytes", fake_analyze_image_bytes)
    monkeypatch.setattr(image_inference.text_analyzer, "hybrid_detector", fake_hybrid_detector)

    response = client.post(
        "/v1/analyze/image/scam",
        files={"file": ("ad.png", b"fake-bytes", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "中等風險"
    assert payload["market_price_source"] == "fallback_local"
    assert payload["is_high_risk_below_market"] is False
    assert payload["score"] == 55.0


def test_analyze_image_scam_invalid_content_type():
    response = client.post(
        "/v1/analyze/image/scam",
        files={"file": ("ad.txt", b"text-bytes", "text/plain")},
    )

    assert response.status_code == 400
    assert "僅支援" in response.json()["detail"]
