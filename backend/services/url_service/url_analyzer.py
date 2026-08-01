from typing import Dict

import joblib
from huggingface_hub import hf_hub_download

from backend.core.config import settings
from backend.utils import features


class Detector():
    def __init__(self):
        self.classifier = None

    def load_model(self):
        # 載入模型
        print("載入 url 推理模型...")
        id = settings.HF_URL_REPO_ID
        token = settings.HF_TOKEN or None
        model_path = hf_hub_download(
            repo_id=id,
            filename="url_scam_classifier.joblib",
            token=token
        )
        self.classifier = joblib.load(model_path)
        print("載入完畢")

    def url_detector(self, url: str) -> Dict:
        if self.classifier is None:
            raise RuntimeError('url模型未載入')

        feat = features.process_url_features(url)
        prob = float(self.classifier.predict_proba(feat)[0][1])
        score = f"{prob:.2f}"
        if prob >= settings.URL_THRESHOLD:
            return {"label": "HIGH", "score": score, "reason": "此鏈結具有風險特徵，請不要點擊前往!"}
        else:
            return {"label": "LOW", "score": score, "reason":  "目前尚未發現鏈結風險，仍需留意陌生鏈結"}

detector = Detector()
