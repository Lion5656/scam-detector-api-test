from typing import Dict
import joblib

from backend.config import settings
from backend.utils import features

class Detector():
    def __init__(self):
        # 載入模型
        self.classifier = joblib.load(f"{settings.BASE_MODEL_PATH}/ml/url_scam_classifier.joblib")

    async def url_detector(self, url: str) -> Dict:
        feat = features.process_url_features(url)
        prob = self.classifier.predict_proba(feat)[0][1]
        print(prob)
        if prob >= 0.55:
            return {"label": "高風險", "score": f"{prob:.2f}"}
        else:
            return {"label": "低風險", "score": f"{prob:.2f}"}

detector = Detector()
