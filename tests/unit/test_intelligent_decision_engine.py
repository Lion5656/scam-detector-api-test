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
        "is_high_risk_below_market": False,
    }
    values.update(overrides)
    return engine.evaluate(**values)


def test_fast_layer_when_product_and_price_are_complete(monkeypatch):
    engine = FusionDecisionEngine()
    monkeypatch.setattr(
        engine,
        "_run_blacklist_tool",
        lambda text: {"hit_keywords": [], "hit_patterns": [], "hit_count": 0},
    )
    monkeypatch.setattr(engine, "_run_rag_tool", lambda text: {"count": 0})

    out = _evaluate(engine)

    assert out["decision_layer"] == "fast"
    assert out["risk_label"] == "低風險"
    assert out["risk_score"] == 25.0


def test_llm_failure_uses_heuristic_fallback(monkeypatch):
    engine = FusionDecisionEngine()
    monkeypatch.setattr(
        engine,
        "_call_llm_deep_analysis",
        lambda product_context, tools: None,
    )
    monkeypatch.setattr(
        engine,
        "_run_blacklist_tool",
        lambda text: {
            "hit_keywords": ["先匯款"],
            "hit_patterns": [],
            "hit_count": 1,
        },
    )
    monkeypatch.setattr(engine, "_run_rag_tool", lambda text: {"count": 1})

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
    assert out["risk_score"] == 31.0
    assert "黑名單命中" in out["reason"]
    assert out["tool_observations"]["blacklist_hit_count"] == 1
    assert out["tool_observations"]["rag_case_count"] == 1


def test_price_risk_starts_from_existing_high_risk_score(monkeypatch):
    engine = FusionDecisionEngine()
    monkeypatch.setattr(
        engine,
        "_call_llm_deep_analysis",
        lambda product_context, tools: None,
    )
    monkeypatch.setattr(
        engine,
        "_run_blacklist_tool",
        lambda text: {"hit_keywords": [], "hit_patterns": [], "hit_count": 0},
    )
    monkeypatch.setattr(engine, "_run_rag_tool", lambda text: {"count": 0})

    out = _evaluate(
        engine,
        selling_price=10000,
        market_price=27900,
        is_high_risk_below_market=True,
    )

    assert out["risk_label"] == "高風險"
    assert out["risk_score"] >= 90.0
