import joblib
from huggingface_hub import hf_hub_download

from backend.core.config import settings
from backend.services.url_service import suffix_policy
from backend.utils import features

MODEL_REASONS = {
    "HIGH": "此鏈結經過判別具有風險特徵，請立即查證網址拼寫是否正確",
    "MEDIUM": "此鏈結可能具有風險，點擊前請先確認來源",
    "LOW": "目前尚未發現鏈結風險，仍需留意陌生鏈結",
}


def get_risk_label(score: float) -> str:
    if score >= settings.URL_HIGH_THRESHOLD:
        return "HIGH"
    if score >= settings.URL_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


class UrlDetector():
    def __init__(self):
        self.classifier = None

    def load_model(self):
        # 載入模型
        print("載入 url 推理模型...")
        id = settings.HF_URL_REPO_ID
        token = settings.HF_TOKEN or None
        model_path = hf_hub_download(
            repo_id=id,
            filename=settings.URL_MODEL_NAME,
            token=token
        )
        self.classifier = joblib.load(model_path)
        print("url模型載入完畢")

    def model_detect(self, url: str, risk_bonus: float = 0.0) -> dict:
        if self.classifier is None:
            raise RuntimeError('url模型未載入')

        feat = features.process_batch_urls([url])
        prob = min(float(self.classifier.predict_proba(feat)[0][1]) + risk_bonus, 1.0)
        label = get_risk_label(prob)
        return {"label": label, "score": f"{prob:.2f}", "reason": MODEL_REASONS[label]}

    def analyze(self, url: str) -> dict:
        """對外入口：短網址與受管制後綴不進模型；可疑 tw 字樣在模型分數上加分後再判斷"""
        signals = suffix_policy.get_suffix_signals(url)

        if signals.is_shortened:
            reason = "此為短網址，無法得知實際導向的網站，點擊前請先確認來源"
            return {"label": "UNKNOWN", "score": 0.0, "reason": reason}

        if signals.is_regulated:
            reason = f"此屬正常網域管制後綴 .{signals.suffix}，註冊需經主管機關審核，仍請確認網址拼寫正確"
            return {"label": "LOW", "score": 0.0, "reason": reason}

        if not signals.has_suspicious_tw:
            return self.model_detect(url)

        result = self.model_detect(url, risk_bonus=settings.URL_TW_RISK_BONUS)
        result["reason"] = f"{result['reason']}，網域名稱中 tw 位置不正常，此類網址常見於假冒台灣單位的網址"
        return result

url_detector = UrlDetector()
