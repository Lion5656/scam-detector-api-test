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
    assert result["score"] == 0.0
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


def test_analyze_shortened_url_skips_model_and_is_medium():
    classifier = StubClassifier(0.99)
    detector = UrlDetector()
    detector.classifier = classifier

    result = detector.analyze("https://reurl.cc/abc123")

    assert result["score"] == f"{settings.URL_SHORTENER_RISK_BONUS:.2f}"
    assert result["label"] == "MEDIUM"
    assert not classifier.called


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
