from typing import Any, cast

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import (AutoTokenizer, Pipeline, PreTrainedTokenizerBase,
                          pipeline)

from backend.core.config import settings


class BaseClassifier:
    """Transformer 模型加載與推理"""

    def __init__(self) -> None:
        self.classifier: Pipeline | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.tochinese: dict[str, str] = {
            "LOW": "低風險",
            "MEDIUM": "中等風險",
            "HIGH": "高風險",
            "UNKNOWN": "未知風險",
        }
        self.rank_map: dict[str, int] = {
            "高風險": 3,
            "中等風險": 2,
            "低風險": 1,
            "未知風險": 0,
        }

    def load_model(self) -> None:
        """加載量化文字分類模型"""
        print("載入量化文字推理模型...")
        model_id = settings.HF_TEXT_REPO_ID
        device = settings.DEVICE
        token = settings.HF_TOKEN or None
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        model = ORTModelForSequenceClassification.from_pretrained(
            model_id,
            file_name="model_quantized.onnx",
            token=token,
            provider="CPUExecutionProvider",
        )
        self.classifier = pipeline(
            "text-classification",
            model=cast(Any, model),
            tokenizer=tokenizer,
            truncation=True,
            device=device,
        )
        self.tokenizer = tokenizer
        print("模型載入完成")

    def _evaluate_distribution(self, result: list[dict[str, Any]]) -> tuple[str, float | None, float]:
        """評估模型輸出的分類分佈"""
        dist: dict[str, float] = {str(item["label"]): float(item["score"]) for item in result}
        sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)

        top_label, top_score = sorted_dist[0]
        second_score = sorted_dist[1][1] if len(sorted_dist) > 1 else 0.0
        margin = top_score - second_score

        if dist.get("UNKNOWN", 0.0) >= settings.UNKNOWN_THRESHOLD:
            return "未知風險", dist.get("UNKNOWN"), margin
        if top_score >= 0.7:
            return self.tochinese.get(top_label, "未知風險"), top_score, margin
        if dist.get("HIGH", 0.0) >= settings.HIGH_THRESHOLD:
            return "高風險", dist.get("HIGH"), margin
        if margin >= 0.1:
            return self.tochinese.get(top_label, "未知風險"), top_score, margin
        return "中等風險", dist.get("MEDIUM"), margin

    def predict_text(self, text: str, max_length: int = 192, stride: int = 64) -> dict[str, Any]:
        """
        對文本進行分類預測
        
        Args:
            text: 輸入文本
            max_length: 最大序列長度
            stride: 滑動窗口步長
            
        Returns:
            預測結果，包含 label, confidence, margin, chunk_consistent
        """
        if self.tokenizer is None:
            raise RuntimeError("分詞器未載入")
        if self.classifier is None:
            raise RuntimeError("文字模型未載入")

        tokens = self.tokenizer.encode(text, truncation=True, add_special_tokens=False)

        if len(tokens) <= max_length:
            result = cast(list[dict[str, Any]], self.classifier(text, top_k=None))
            label, confidence, margin = self._evaluate_distribution(result)
            return {
                "label": label,
                "confidence": confidence,
                "margin": margin,
                "chunk_consistent": True,
            }

        chunks: list[str] = []
        for i in range(0, len(tokens), max_length - stride):
            chunk_tokens = tokens[i : i + max_length]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

        results = cast(
            list[list[dict[str, Any]]],
            self.classifier(chunks, truncation=True, max_length=max_length, top_k=None),
        )

        final_label = "未知風險"
        final_confidence = 0.0
        final_margin = 0.0
        chunk_labels: list[str] = []

        for chunk_result in results:
            label, confidence, margin = self._evaluate_distribution(chunk_result)
            chunk_labels.append(label)
            current_rank = self.rank_map.get(label, 0)
            final_rank = self.rank_map.get(final_label, 0)

            if current_rank > final_rank:
                final_label = label
                final_confidence = float(confidence or 0.0)
                final_margin = margin
            elif current_rank == final_rank:
                final_confidence = max(final_confidence, float(confidence or 0.0))
                final_margin = max(final_margin, margin)

        return {
            "label": final_label,
            "confidence": final_confidence,
            "margin": final_margin,
            "chunk_consistent": len(set(chunk_labels)) <= 2,
        }


base_classifier = BaseClassifier()
