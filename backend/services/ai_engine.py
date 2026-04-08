import os
from typing import Any, Dict, List, Tuple, cast

from fastapi import HTTPException
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, Pipeline, pipeline

from backend.config import settings
from backend.schemas.analysis import Request, Response
from backend.utils.pattern import compare_rules
from backend.utils.text_cleaner import normalize_text


class InferenceEngine():
    def __init__(self) -> None:
        self.classifier: Pipeline | None = None # 模型分類器, 初始未載入 = None
        self.tochinese: Dict[str, str] = {
            "LOW": "低風險",
            "MEDIUM": "中等風險",
            "HIGH": "高風險", 
            "UNKNOWN": "未知"
        }
        self.model_score_map: Dict[str, float] = {
            "高風險": 100,
            "中等風險": 50,
            "低風險": 20
        }

    # 加載模型
    def load_model(self):
        print("載入量化Bert ONNX推理模型...")
        id = settings.HF_REPO_ID
        device = settings.DEVICE
        token = os.getenv("HF_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(id, token=token)
        model = ORTModelForSequenceClassification.from_pretrained(id, file_name="model_quantized.onnx", token=token, provider="CPUExecutionProvider")
        self.classifier = pipeline("text-classification", model=cast(Any, model), tokenizer=tokenizer, truncation=True, device=device)


    # 模型預測 + 決策分類
    def _predict(self, text: str) -> Tuple[str | None, float | None]:
        if self.classifier is None:
            raise RuntimeError("模型未載入")

        assert self.classifier is not None

        results =  cast(List[Dict[str, Any]], self.classifier(text, top_k=None))
        # 如果模型輸出是多層列表，取第一層 [[{"label": "HIGH", "score": 0.95}, ...]]
        if isinstance(results[0], list):
            results = results[0]
        dist: Dict[str, float] = {str(item['label']): float(item['score']) for item in results} # type: ignore

        # 排序標籤分數由大到小
        sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)

        top_label, top_score = sorted_dist[0]
        second_score = sorted_dist[1][1]
        margin = top_score - second_score

        if dist.get("UNKNOWN", 0.0) >= 0.7:
            return "未知", dist.get("UNKNOWN")
        if top_score >= 0.7:
            return self.tochinese.get(top_label), top_score
        if dist.get("HIGH", 0.0) >= 0.5:
            return "高風險", dist.get("HIGH")
        if margin >= 0.1:
            return self.tochinese.get(top_label), top_score
        return "中等風險", dist.get("MEDIUM")


    # 正則模糊比對 + 模型預測
    async def cascaded_detector(self, req: Request) -> Response:
        text = req.text
        
        if not text:
            raise HTTPException(status_code=400, detail="text不能為空白")
        
        # 分配權重
        w_model = settings.MODEL_WEIGHT
        w_rule = settings.REGEX_WEIGHT

        text = normalize_text(text)
        # 正則比對
        rule_score, rule_reason, rule_hit = compare_rules(text)
        # 模型預測
        model_label , model_conf_score = self._predict(text)

        model_conf_score = 0 if model_conf_score is None else model_conf_score

        # 檢測反詐關鍵詞
        extra_reason = rule_reason if "反詐" in rule_reason else ""

        # 命中詞去重
        rule_hit = set(rule_hit)

        # 特別情境1. 如果模型判斷為未知風險且命中規則 = 0，判定未知
        if model_label == "未知風險" and len(rule_hit) == 0:
            return Response(label="未知風險", score="未知", confidence_score=model_conf_score, reason="語句缺乏明確資訊，無法進行有效判斷，評估風險為未知")
        # 特別情境2. 如過模型預測很正常 (低風險&信心分數 > 0.8 且 命中關鍵字 <= 2)，判定低風險
        if model_label == "低風險" and model_conf_score >= 0.8:
                if len(rule_hit) <= 2:
                    return Response(label="低風險", score=10.0, confidence_score=model_conf_score, reason=f"此訊息所含詐騙特徵較少{extra_reason}, 評估風險為低。")
        # 特別情境3. 如過模型預測高風險且命中規則 >= 1，判定高風險
        if model_label == "高風險" and len(rule_hit) >= 1:
            return Response(label="高風險", score=85.0, confidence_score=model_conf_score, reason=f"此訊息{rule_reason}, 評估風險為高")
        # 特別情境4. 如果比對規則命中 >= 5，直接判定為高風險
        if len(rule_hit) >= 5:
            return Response(label="高風險", score=95.0, confidence_score=model_conf_score, reason=f"此訊息{rule_reason}, 評估風險為高")
        
        # 規則基本分數
        rule_base = 20
        # 一般情況
        if model_label == "高風險":
            model_eval = 100
        elif model_label == "中等風險":
            model_eval = 50
        elif model_label == "低風險":
            model_eval = 20

        final_score = max(rule_score, (rule_base + rule_score) * w_rule + model_eval * w_model)

        if final_score >= 80:
            return Response(label="高風險", score=final_score, confidence_score=model_conf_score, reason=f"此訊息{rule_reason}, 評估風險為高")
        if final_score >= 40 and rule_reason:
            return Response(label="中等風險", score=final_score, confidence_score=model_conf_score, reason=f"此訊息{rule_reason}, 評估風險為中等")
        if final_score >= 40:
            return Response(label="中等風險", score=final_score, confidence_score=model_conf_score, reason="此訊息疑似有詐騙風險，但特徵較模糊，評估風險為中等")
        else:
            return Response(label="低風險", score=final_score, confidence_score=model_conf_score, 
                            reason=f"此訊息所含詐騙特徵較少{extra_reason}, 評估風險為低。")

    # only模型預測
    async def model_detector(self, req: Request) -> Response:
        text = normalize_text(req.text)
        if not text:
            raise HTTPException(status_code=400, detail="text 不能為空白")
        label, conf_score =  self._predict(text)
        return Response(label=label or "",  confidence_score=conf_score)
    
inference_engine = InferenceEngine()
