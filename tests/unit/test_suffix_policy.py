import pytest

from backend.core.config import settings
from backend.services.url_service.suffix_policy import get_suffix_signals
from backend.services.url_service.url_analyzer import UrlDetector


@pytest.mark.parametrize("url", [
    "https://www.moe.gov.tw/news",
    "https://www.ntu.edu.tw",
    "https://www.mnd.mil.tw",
    "https://www.example.com.tw",
    "https://WWW.NHI.GOV.TW.:443/",
])
def test_regulated_suffix_is_whitelisted(url):
    assert get_suffix_signals(url).is_regulated


@pytest.mark.parametrize("url", [
    "https://fake-gov.tw",
    "https://abc.game.tw",
    "https://gov.tw.login-check.xyz",
    "https://x.gov.tw@evil.xyz/login",
    "https://harvard.edu",
    "http://192.168.1.1/",
])
def test_unregulated_suffix_is_not_whitelisted(url):
    assert not get_suffix_signals(url).is_regulated


@pytest.mark.parametrize("url, expected", [
    # 後綴是 tw 但不在白名單
    ("https://ezpay.tw", True),
    ("https://abc.game.tw", True),
    # tw 黏在網域主體
    ("https://ezpaytw.com", True),
    ("https://nhi_tw88.cc", True),
    # tw 不在後綴位置
    ("https://gov.tw.login-check.xyz", True),
    ("https://tw-post.top", True),
    # COMMON_DOMAINS 豁免
    ("https://shopee.tw", False),
    ("https://tw.yahoo.com", False),
    # 無 tw 或只是 tw 開頭
    ("https://www.twitter.com", False),
    ("https://www.twse.com.tw", False),
    ("https://example.com", False),
])
def test_suspicious_tw(url, expected):
    assert get_suffix_signals(url).has_suspicious_tw is expected


@pytest.mark.parametrize("url, expected", [
    ("https://reurl.cc/abc123", True),
    ("https://lihi1.com/xyz", True),
    ("https://bit.ly/3AbCdE", True),
    ("https://preview.tinyurl.com/abc", True),
    ("https://x.gov.tw@bit.ly/abc", True),
    ("https://bit.ly.evil.xyz/abc", False),
    ("https://notbit.ly/abc", False),
    ("https://youtu.be/abc", False),
    ("https://www.moe.gov.tw", False),
])
def test_shortened_url(url, expected):
    assert get_suffix_signals(url).is_shortened is expected


class StubClassifier:
    def __init__(self, prob):
        self.prob = prob
        self.called = False

    def predict_proba(self, _features):
        self.called = True
        return [[1 - self.prob, self.prob]]


def make_detector(prob):
    detector = UrlDetector()
    detector.classifier = StubClassifier(prob)
    return detector


def test_analyze_regulated_suffix_skips_model():
    classifier = StubClassifier(0.99)
    detector = UrlDetector()
    detector.classifier = classifier

    result = detector.analyze("https://www.moe.gov.tw/news")

    assert result["label"] == "LOW"
    assert result["score"] == "0.00"
    assert not classifier.called


def test_analyze_suspicious_tw_adds_bonus_and_relabels():
    prob = settings.URL_HIGH_THRESHOLD - 0.01
    detector = make_detector(prob)

    result = detector.analyze("https://tw-post.top/verify")

    assert result["score"] == f"{prob + settings.URL_TW_RISK_BONUS:.2f}"
    assert result["label"] == "HIGH"
    assert "tw" in result["reason"]


def test_analyze_bonus_is_capped_at_one():
    detector = make_detector(0.95)

    result = detector.analyze("https://ezpay.tw")

    assert result["score"] == "1.00"


def test_analyze_shortened_url_analyzes_resolved_target():
    classifier = StubClassifier(0.99)
    detector = UrlDetector(resolve_redirect=lambda url: "https://tw-post.top/verify")
    detector.classifier = classifier

    result = detector.analyze("https://reurl.cc/abc123")

    assert classifier.called
    assert result["label"] == "HIGH"
    assert result["reason"].startswith("此短網址實際導向 https://tw-post.top/verify")


def test_analyze_unresolved_shortened_url_is_unknown():
    classifier = StubClassifier(0.99)
    detector = UrlDetector(resolve_redirect=lambda url: None)
    detector.classifier = classifier

    result = detector.analyze("https://reurl.cc/abc123")

    assert result == {"label": "UNKNOWN", "score": "0.00", "reason": result["reason"]}
    assert "短網址" in result["reason"]
    assert not classifier.called


def test_analyze_shortened_url_expands_only_once():
    calls = []

    def resolve(url):
        calls.append(url)
        return "https://bit.ly/next"

    detector = UrlDetector(resolve_redirect=resolve)
    detector.classifier = StubClassifier(0.99)

    result = detector.analyze("https://reurl.cc/abc123")

    assert calls == ["https://reurl.cc/abc123"]
    assert result["label"] == "UNKNOWN"


@pytest.mark.parametrize(
    "url", ["https://www.moe.gov.tw", "https://www.ntu.edu.tw", "https://www.mnd.mil.tw"]
)
def test_restricted_suffix_signal(url):
    signals = get_suffix_signals(url)
    assert signals.is_regulated and signals.is_restricted


@pytest.mark.parametrize(
    "url", ["https://www.example.com.tw", "https://myname.idv.tw", "https://a.org.tw"]
)
def test_registered_suffix_is_regulated_but_not_restricted(url):
    signals = get_suffix_signals(url)
    assert signals.is_regulated and not signals.is_restricted


def test_analyze_registered_suffix_runs_model_and_caps_at_medium():
    classifier = StubClassifier(0.99)
    detector = UrlDetector()
    detector.classifier = classifier

    result = detector.analyze("https://abc.com.tw/login/verify-account")

    assert classifier.called
    assert result["label"] == "MEDIUM"
    assert result["score"] == f"{settings.URL_HIGH_THRESHOLD - 0.01:.2f}"
    assert "遭入侵或冒名申請" in result["reason"]


def test_analyze_registered_suffix_low_risk_keeps_model_score():
    detector = make_detector(0.05)

    result = detector.analyze("https://www.example.com.tw")

    assert result["label"] == "LOW"
    assert result["score"] == "0.05"
    assert "實名註冊後綴" in result["reason"]


@pytest.mark.parametrize("url, suffix_flag, host_flag", [
    ("https://dcard.tw", True, False),
    ("https://abc.game.tw", True, False),
    ("https://nhi-gov-tw.com", False, True),
    ("https://gov.tw.login-check.xyz", False, True),
    ("https://shopee.tw", False, False),
])
def test_tw_signal_types_are_separated(url, suffix_flag, host_flag):
    signals = get_suffix_signals(url)
    assert signals.has_unregulated_tw_suffix is suffix_flag
    assert signals.has_tw_in_host is host_flag


def test_analyze_tw_reason_matches_signal_type():
    detector = make_detector(0.1)

    suffix_reason = detector.analyze("https://dcard.tw")["reason"]
    host_reason = detector.analyze("https://nhi-gov-tw.com")["reason"]

    assert "未受管制的 .tw 後綴" in suffix_reason and "夾帶 tw" not in suffix_reason
    assert "夾帶 tw" in host_reason and "未受管制" not in host_reason


@pytest.mark.parametrize("prob, expected", [
    (settings.URL_HIGH_THRESHOLD, "HIGH"),
    (settings.URL_MEDIUM_THRESHOLD, "MEDIUM"),
    (settings.URL_MEDIUM_THRESHOLD - 0.01, "LOW"),
])
def test_model_detect_risk_tiers(prob, expected):
    detector = make_detector(prob)

    assert detector.model_detect("https://example.com")["label"] == expected


def test_analyze_common_domain_gets_no_bonus():
    detector = make_detector(0.3)

    result = detector.analyze("https://shopee.tw")

    assert result == detector.model_detect("https://shopee.tw")
