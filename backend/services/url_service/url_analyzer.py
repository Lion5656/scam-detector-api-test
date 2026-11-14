from collections.abc import Callable

import joblib
from huggingface_hub import hf_hub_download

from backend.core.config import settings
from backend.services.url_service import suffix_policy
from backend.services.url_service.redirect_resolver import resolve_final_url
from backend.utils import features

MODEL_REASONS = {
    "HIGH": "此鏈結經過判別具有風險特徵，請立即查證網址拼寫是否正確",
    "MEDIUM": "此鏈結可能具有風險，點擊前請先確認來源",
    "LOW": "目前尚未發現鏈結風險，仍需留意陌生鏈結",
}
SHORTENED_UNRESOLVED_REASON = "此為短網址，無法得知實際導向的網站，點擊前請先確認來源"
MAX_DISPLAY_URL_LENGTH = 100


def get_risk_label(score: float) -> str:
    if score >= settings.URL_HIGH_THRESHOLD:
        return "HIGH"
    if score >= settings.URL_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def format_score(score: float) -> str:
    """所有路徑的 score 一律輸出兩位小數字串，維持 API 回傳型別一致。"""
    return f"{score:.2f}"


class UrlDetector:
    def __init__(self, resolve_redirect: Callable[[str], str | None] = resolve_final_url):
        self.classifier = None
        self._resolve_redirect = resolve_redirect

    def load_model(self):
        print("載入 url 推理模型...")
        model_path = hf_hub_download(
            repo_id=settings.HF_URL_REPO_ID,
            filename=settings.URL_MODEL_NAME,
            token=settings.HF_TOKEN or None,
        )
        self.classifier = joblib.load(model_path)
        print("url模型載入完畢")

    def model_detect(self, url: str, risk_bonus: float = 0.0, max_score: float = 1.0) -> dict:
        if self.classifier is None:
            raise RuntimeError('url模型未載入')

        feat = features.process_batch_urls([url])
        prob = float(self.classifier.predict_proba(feat)[0][1])
        score = min(prob + risk_bonus, max_score, 1.0)
        label = get_risk_label(score)
        return {"label": label, "score": format_score(score), "reason": MODEL_REASONS[label]}

    def analyze(self, url: str, *, expand_shortened: bool = True) -> dict:
        """對外入口：依網域訊號決定是否進模型、加減分與補充理由。

        - 短網址：展開後分析實際導向的網址，無法展開時回 UNKNOWN
        - gov.tw／edu.tw／mil.tw：僅限特定單位註冊，不進模型直接判低風險
        - com.tw 等實名註冊後綴：仍進模型，但最高只到中等風險
        - 可疑 tw 字樣：模型分數加分，並依訊號類型補充理由
        """
        signals = suffix_policy.get_suffix_signals(url)
        if signals.is_shortened:
            return self._analyze_shortened(url, expand_shortened)

        if signals.is_restricted:
            reason = f"此屬政府、學校或軍方專用後綴 .{signals.suffix}，僅限該類單位註冊，仍請確認網址拼寫正確"
            return {"label": "LOW", "score": format_score(0.0), "reason": reason}

        risk_bonus = settings.URL_TW_RISK_BONUS if signals.has_suspicious_tw else 0.0
        # 實名註冊不代表網站本身安全（可能遭入侵或冒名申請），因此仍跑模型，只是不判到高風險
        max_score = settings.URL_HIGH_THRESHOLD - 0.01 if signals.is_regulated else 1.0
        result = self.model_detect(url, risk_bonus=risk_bonus, max_score=max_score)

        notes: list[str] = []
        if signals.is_regulated and result["label"] == "LOW":
            notes.append(f"此網域為 .{signals.suffix} 實名註冊後綴")
        elif signals.is_regulated:
            notes.append(f"雖為 .{signals.suffix} 實名註冊後綴，仍可能是遭入侵或冒名申請的網站")
        if signals.has_unregulated_tw_suffix:
            notes.append(f"此網域使用未受管制的 .{signals.suffix} 後綴，任何人皆可註冊，常見於假冒台灣企業的網址")
        if signals.has_tw_in_host:
            notes.append("網域名稱中夾帶 tw 字樣，此類網址常見於假冒台灣單位的網址")

        if notes:
            result["reason"] = "，".join([result["reason"], *notes])
        return result

    def _analyze_shortened(self, url: str, expand: bool) -> dict:
        final_url = None
        if expand and settings.URL_EXPAND_SHORTENED:
            final_url = self._resolve_redirect(url)
        if not final_url:
            score = format_score(0.0)
            return {"label": "UNKNOWN", "score": score, "reason": SHORTENED_UNRESOLVED_REASON}

        # 只展開一層，避免短網址互相導向造成遞迴
        result = self.analyze(final_url, expand_shortened=False)
        display_url = final_url
        if len(display_url) > MAX_DISPLAY_URL_LENGTH:
            display_url = f"{display_url[:MAX_DISPLAY_URL_LENGTH]}..."
        result["reason"] = f"此短網址實際導向 {display_url}，{result['reason']}"
        return result


url_detector = UrlDetector()
