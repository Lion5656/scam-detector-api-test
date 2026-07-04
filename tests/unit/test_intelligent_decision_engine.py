from backend.services.image_service.intelligent_decision_engine import IntelligentDecisionEngine


def test_fast_layer_when_confident():
    engine = IntelligentDecisionEngine()
    text_result = {
        "label": "低風險",
        "score": 12.0,
        "reason": "特徵低",
        "cls_model_confidence": 0.93,
    }

    out = engine.evaluate(text_result, "Apple iPhone 15", "Apple iPhone 15", "一般交易通知")

    assert out["decision_layer"] == "fast"
    assert out["risk_label"] == "低風險"


def test_llm_simulated_layer_when_uncertain(monkeypatch):
    engine = IntelligentDecisionEngine()

    monkeypatch.setattr(engine, "_call_llm_deep_analysis", lambda text_result, tools: None)
    monkeypatch.setattr(engine, "_run_blacklist_tool", lambda text: {"hit_keywords": ["先匯款"], "hit_patterns": [], "hit_count": 1})
    monkeypatch.setattr(engine, "_run_rag_tool", lambda text: {"count": 1, "snippets": ["相似詐騙案例"]})
    monkeypatch.setattr(engine, "_run_online_price_tool", lambda brand_model, product_name: {"query": brand_model, "price": 32000})

    text_result = {
        "label": "中等風險",
        "score": 45.0,
        "reason": "資訊不足",
        "cls_model_confidence": 0.51,
    }

    out = engine.evaluate(text_result, "未知商品", "未知型號", "請先匯款再出貨")

    assert out["decision_layer"] == "llm_simulated"
    assert out["risk_score"] >= 45.0
    assert "黑名單命中" in out["reason"]
    assert out["tool_observations"]["blacklist_hit_count"] == 1
