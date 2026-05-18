import os
from typing import Dict

import joblib
from huggingface_hub import hf_hub_download

from backend.config import settings
from backend.utils import features

class Detector():
    def __init__(self):
        self.classifier = None

    def load_model(self):
        # 載入模型
        print("載入 url 推理模型...")
        id = settings.HF_URL_REPO_ID
        token = os.getenv("HF_TOKEN")
        model_path = hf_hub_download(
            repo_id=id,
            filename="url_scam_classifier.joblib",
            token=token
        )
        self.classifier = joblib.load(model_path)
        print("載入完畢")

    async def url_detector(self, url: str) -> Dict:
        if self.classifier is None:
            raise RuntimeError('url模型未載入')

        feat = features.process_url_features(url)
        prob = float(self.classifier.predict_proba(feat)[0][1])
        score = round(prob, 2)
        print(prob)
        if prob >= 0.46:
            return {"label": "詐騙", "score": score}
        else:
            return {"label": "安全", "score": score}

detector = Detector()
