from backend.services.image_price_service.risk.fusion_decision_engine import (
    FusionDecisionEngine,
)


def _evaluate(engine: FusionDecisionEngine, **overrides):
    values = {
        "product_name": "Apple iPhone 15",
        "brand_model": "Apple iPhone 15",
        "text": "正常商品資訊",
        "selling_price": 25000,
        "market_price": 27900,
        "market_price_source": "online",
    }
    values.update(overrides)
    return engine.evaluate(**values)


def test_fast_layer_when_product_and_price_are_complete(monkeypatch):
    engine = FusionDecisionEngine()
    monkeypatch.setattr(
        engine,
        "_run_blacklist_hit",
        lambda text: 0,
    )

    out = _evaluate(engine)

    assert out["decision_layer"] == "fast"
    assert out["risk_label"] == "LOW"
    assert out["risk_score"] == 20.0
    assert out["market_price_source"] == "online"


def test_llm_failure_uses_heuristic_fallback(monkeypatch):
    engine = FusionDecisionEngine()
    monkeypatch.setattr(
        engine,
        "_call_llm_deep_analysis",
        lambda product_context, tools: None,
    )
    monkeypatch.setattr(
        engine,
        "_run_blacklist_hit",
        lambda text: 1,
    )

    out = _evaluate(
        engine,
        product_name="未知商品",
        brand_model="未知型號",
        text="請先匯款再出貨",
        selling_price=0,
        market_price=0,
        market_price_source="fallback_local",
    )

    assert out["decision_layer"] == "llm_simulated"
    assert out["risk_score"] == 20.0
    assert out["market_price_source"] == "fallback_local"
    assert "黑名單命中" in out["reason"]


def test_price_risk_starts_from_existing_high_risk_score(monkeypatch):
    engine = FusionDecisionEngine()
    monkeypatch.setattr(
        engine,
        "_call_llm_deep_analysis",
        lambda product_context, tools: None,
    )
    monkeypatch.setattr(
        engine,
        "_run_blacklist_hit",
        lambda text: 0,
    )

    out = _evaluate(
        engine,
        selling_price=10000,
        market_price=27900,
    )

    assert out["risk_label"] == "HIGH"
    assert out["risk_score"] >= 90.0
    assert "低於行情 50% 規則觸發" in out["evidence"]


def test_price_risk_boundary_rules_are_owned_by_fusion_engine():
    assert FusionDecisionEngine._has_price_risk(13_949, 27_900) is True
    assert FusionDecisionEngine._has_price_risk(13_950, 27_900) is False
    assert FusionDecisionEngine._has_price_risk(55_799, 27_900) is False
    assert FusionDecisionEngine._has_price_risk(55_800, 27_900) is True
    assert FusionDecisionEngine._has_price_risk(0, 27_900) is False


def test_price_rule_overrides_successful_llm_low_risk_result(monkeypatch):
    engine = FusionDecisionEngine()
    monkeypatch.setattr(engine, "_run_blacklist_hit", lambda text: 0)
    monkeypatch.setattr(
        engine,
        "_call_llm_deep_analysis",
        lambda product_context, tools: {
            "risk_label": "LOW",
            "risk_score": 25.0,
            "reason": "模型判定為低風險",
            "evidence": ["模型證據"],
            "confidence": 0.8,
            "decision_layer": "llm",
        },
    )

    out = _evaluate(
        engine,
        selling_price=55_800,
        market_price=27_900,
    )

    assert out["risk_label"] == "HIGH"
    assert out["risk_score"] == 90.0
    assert "模型判定為低風險" in out["reason"]
    assert "高於行情 2 倍規則觸發" in out["evidence"]
    assert out["decision_layer"] == "llm"
