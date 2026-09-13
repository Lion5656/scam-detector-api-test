"""URL 特徵抽取，供 service 端呼叫"""

import math
import re
from functools import lru_cache
from urllib.parse import parse_qs, urlparse, urlunsplit

import pandas as pd
import tldextract
from pandas import DataFrame

FEATURE_NAMES = [
    "url_length", "num_dot", "num_dash", "num_slash",
    "is_https", "domain_length", "subdomain_depth", "has_digit_in_domain",
    "num_dash_in_domain", "num_phishing_keywords", "has_tw_keyword",
    "path_length", "num_path_segments", "num_params",
    "has_utm", "has_encoded_chars", "is_suspicious_tld",
    "domain_entropy", "is_real_gov", "is_fake_gov",
    "is_common_domain", "is_shortened", "is_common_tld",
    "num_keywords_in_domain", "num_keywords_in_path",
    "brand_in_domain_not_own", "domain_token_count",
    "min_edit_distance_to_brand", "is_punycode",
]

N_FEATURES = 29

SUSPICIOUS_TLD = ["xyz", "top", "cc", "tk", "ml", "ga", "club", "vip"]

TW_KEYWORDS = [
    "gov", "nhia", "健保", "郵局",
    "post", "etax", "tax", "電費",
    "taipower", "7-11", "familymart", "物流",
]

SHORTENER = ["bit.ly", "t.co", "tinyurl.com", "t.ly"]

# 常見網域白名單
COMMON_DOMAINS = {
    "google.com", "youtube.com", "facebook.com",
    "amazon.com", "wikipedia.org", "microsoft.com",
    "x.com", "yahoo.com", "shopee.tw"
}

COMMON_TLD = ["com", "org", "net", "tw", "com.tw"]

PHISHING_KEYWORDS = [
    "login", "verify", "secure", "confirm", "account",
    "authenticate", "payment", "receipt", "identity",
    "update", "exchange", "security", "auth", "signin",
]

GOV_TOKENS = ["gov", "govtw", "egov", "nhia", "nhi", "etax", "moi", "npa", "g0v", "g0vtw"]

GOV_SUFFIX = {"gov.tw", "gov", "gov.uk", "go.jp"}

# 只放網域主體那一段（不含後綴），判斷的是「這個詞出現在誰的註冊網域裡」。
BRANDS = frozenset({
    "apple", "icloud", "paypal", "google", "gmail", "youtube", "facebook",
    "instagram", "whatsapp", "microsoft", "outlook", "office365", "onedrive",
    "amazon", "netflix", "linkedin", "twitter", "dropbox", "adobe", "docusign",
    "steam", "roblox", "discord", "telegram", "signal", "zoom", "spotify",
    "booking", "airbnb", "alibaba", "aliexpress", "wise", "revolut",
    "fedex", "dhl", "ups", "usps", "hermes", "yamato", "sfexpress",
    "binance", "coinbase", "metamask", "trustwallet", "ledger", "kraken",
    "bitfinex", "okx", "bybit",
    "shopee", "pchome", "momoshop", "ruten", "yahoo", "line", "gamer",
    "esunbank", "cathaybk", "ctbcbank", "fubon", "megabank", "taishinbank",
    "firstbank", "landbank", "hncb", "sinopac", "chb", "tcb",
    "chunghwa", "cht", "taipower", "nhi", "post", "ubot", "jkos", "icash",
})

# 形近字元替換是常見的躲避字串比對手法，比對前先正規化。
HOMOGLYPHS = str.maketrans({
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
    "@": "a", "$": "s",
})

MAX_EDIT_DISTANCE = 5   # 超過就沒有 typosquat 的意義，截斷以壓縮值域
MIN_TOKEN_LEN = 4       # 短詞之間的編輯距離沒有意義
BRAND_PREFIX_SLACK = 4  # 品牌當 token 前綴時，後面最多還能接幾個字元

# 會被前綴規則誤判成品牌的一般英文字（revolut+ion、post+al、line+age…）。
NOT_BRAND_TOKENS = frozenset({
    "revolution", "revolutions", "revolutionary",
    "postal", "posted", "poster", "posters", "posting", "postre",
    "lineage", "linear", "lineup", "liner", "liners",
    "upstream", "upside", "upsell", "upstate",
    "signals", "signage", "signature",
    "steamer", "steamed", "wisely", "wisdom",
    "zoomed", "zoomer", "adobers",
    "amazonas", "amazonia",
})

_LONG_BRANDS = frozenset(b for b in BRANDS if len(b) >= MIN_TOKEN_LEN)


def _keyword_regex(keywords):
    """詞界比對，避免 author 命中 auth、paypal 命中 pay；但允許複數形。"""
    alt = "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9])({alt})s?(?![a-z])", re.IGNORECASE)


PHISHING_KEYWORD_RE = _keyword_regex(PHISHING_KEYWORDS)
TW_KEYWORD_RE = _keyword_regex([k for k in TW_KEYWORDS if k.isascii()])
TW_KEYWORD_CJK = [k for k in TW_KEYWORDS if not k.isascii()]

# 這裡刻意用子字串而非詞界：把政府字樣黏進網域正是攻擊手法本身
# （evilgov.tw、nhigovtw.com），加詞界反而會漏掉。
GOV_TOKEN_RE = re.compile("|".join(re.escape(k) for k in GOV_TOKENS), re.IGNORECASE)


def count_keywords(text: str, pattern: re.Pattern) -> int:
    """命中的不同關鍵詞數量，同一個詞出現多次只算一次。"""
    return len({m.group(1).lower() for m in pattern.finditer(text)})


def calc_entropy(s):
    if not s:
        return 0
    prob = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob)


def _typo_threshold(brand: str) -> int:
    """短品牌不能用大門檻，否則 wise↔nibe 距離 2、line↔nine 距離 1 都會命中。"""
    if len(brand) < 5:
        return 0
    return 1 if len(brand) == 5 else 2


def brand_hit(tokens: tuple[str, ...]) -> str | None:
    """品牌詞是否出現在網域裡，回傳命中的品牌。

    只認 token 邊界，不做純子字串比對（否則 online 會命中 line、
    groovypost 會命中 post）。品牌本身是完整 token，或是某個 token 的前綴
    且後面最多再接 BRAND_PREFIX_SLACK 個字元，才算命中。
    """
    token_set = set(tokens)
    for b in BRANDS:
        if b in token_set:
            return b
        for t in tokens:
            if t in NOT_BRAND_TOKENS:
                continue
            if t.startswith(b) and 0 < len(t) - len(b) <= BRAND_PREFIX_SLACK:
                return b
    return None


@lru_cache(maxsize=200_000)
def edit_distance(a: str, b: str, cap: int = MAX_EDIT_DISTANCE) -> int:
    """Levenshtein 距離，超過 cap 提早結束。"""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap
        prev = cur
    return min(prev[-1], cap)


def _domain_tokens(label: str) -> list[str]:
    """把註冊網域主體切成詞：連字號、底線、數字邊界都算分隔。"""
    return [t for t in re.split(r"[-_]+|(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", label) if t]


@lru_cache(maxsize=100_000)
def brand_signals(host_label: str, subdomain: str) -> tuple[int, int]:
    """回傳 (brand_in_domain_not_own, min_edit_distance_to_brand)。

    brand_in_domain_not_own：品牌詞出現在 host 裡但註冊主體不是該品牌。
        apple.com -> 0，secure-login-apple-id.xyz -> 1
    min_edit_distance_to_brand：主體到最近品牌的距離，0 = 就是品牌本人，
        1-2 = typosquat，>=3 視為無關。
    """
    label = host_label.lower()
    normalized = label.translate(HOMOGLYPHS)

    if label in BRANDS or normalized in BRANDS:
        return 0, 0

    host = f"{subdomain}.{label}".lower() if subdomain else label
    host_norm = host.translate(HOMOGLYPHS)
    tokens = tuple(t for part in host_norm.split(".") for t in _domain_tokens(part))
    in_domain = int(brand_hit(tokens) is not None)

    # 兩邊都要求至少 MIN_TOKEN_LEN 字元，否則 tw / id 這種短詞會產生無意義的小距離。
    candidates = [c for c in (normalized, *_domain_tokens(normalized))
                  if len(c) >= MIN_TOKEN_LEN]
    best = MAX_EDIT_DISTANCE
    for c in candidates:
        for b in _LONG_BRANDS:
            d = edit_distance(c, b)
            if d <= _typo_threshold(b):
                best = min(best, d)
    return in_domain, best


def strip_www(host: str) -> str:
    return host[4:] if host.lower().startswith("www.") else host


def subdomain_depth_of(url: str) -> int:
    """子網域層數，www. 不算。與 subdomain_depth 特徵同一套定義。"""
    host = strip_www(urlparse(url).netloc)
    subdomain = tldextract.extract(host).subdomain
    return len(subdomain.split(".")) if subdomain else 0


def extract_features(url: str) -> dict:
    """抽出單一 URL 的 29 個特徵，回傳 dict。"""
    parsed = urlparse(url)
    domain = strip_www(parsed.netloc)
    path = parsed.path
    query = parsed.query
    scheme = parsed.scheme
    ext = tldextract.extract(domain)
    root_domain = f"{ext.domain}.{ext.suffix}"
    subdomain = ext.subdomain
    tld = ext.suffix

    features = {}

    # 全部算在剝掉 www. 之後的 URL 上，否則 www. 會讓長度與點數憑空變動。
    url_norm = urlunsplit((scheme, domain, path, query, parsed.fragment))
    url_lower = url_norm.lower()

    features["url_length"] = len(url_norm)
    features["num_dot"] = url_norm.count(".")
    features["num_dash"] = url_norm.count("-")
    features["num_slash"] = url_norm.count("/")
    features["is_https"] = int(scheme == "https")

    features["domain_length"] = len(domain)
    features["subdomain_depth"] = len(subdomain.split(".")) if subdomain else 0
    features["has_digit_in_domain"] = int(bool(re.search(r"\d", domain)))
    features["num_dash_in_domain"] = domain.count("-")

    features["num_phishing_keywords"] = count_keywords(url_lower, PHISHING_KEYWORD_RE)
    features["has_tw_keyword"] = int(
        bool(TW_KEYWORD_RE.search(url_lower))
        or any(k in url_norm for k in TW_KEYWORD_CJK)
    )

    features["path_length"] = len(path)
    features["num_path_segments"] = len([p for p in path.split("/") if p])

    params = parse_qs(query)
    features["num_params"] = len(params)
    features["has_utm"] = int("utm_" in query)
    features["has_encoded_chars"] = int("%" in url_norm)

    features["is_suspicious_tld"] = int(tld in SUSPICIOUS_TLD)
    features["domain_entropy"] = calc_entropy(domain)

    # 必須比對「後綴」，不能用 endswith，否則 fake-gov.tw、evilgov.tw 會被當真政府。
    features["is_real_gov"] = int(tld in GOV_SUFFIX)

    host_part = f"{subdomain}.{ext.domain}" if subdomain else ext.domain
    features["is_fake_gov"] = int(
        bool(GOV_TOKEN_RE.search(host_part.lower())) and tld not in GOV_SUFFIX
    )

    features["is_common_domain"] = int(root_domain in COMMON_DOMAINS)
    features["is_shortened"] = int(root_domain in SHORTENER)
    features["is_common_tld"] = int(tld in COMMON_TLD)

    # 關鍵字在網域 vs 在 path 是兩種強度不同的訊號，分開算。
    features["num_keywords_in_domain"] = count_keywords(
        host_part.lower(), PHISHING_KEYWORD_RE
    )
    features["num_keywords_in_path"] = count_keywords(
        f"{path}?{query}".lower(), PHISHING_KEYWORD_RE
    )

    in_domain, brand_dist = brand_signals(ext.domain, subdomain)
    features["brand_in_domain_not_own"] = in_domain
    features["min_edit_distance_to_brand"] = brand_dist

    features["domain_token_count"] = len(_domain_tokens(ext.domain.lower()))
    features["is_punycode"] = int("xn--" in domain.lower())

    return features


def process_batch_urls(urls: list) -> DataFrame:
    """把一批 URL 轉成特徵矩陣，欄位順序固定為 FEATURE_NAMES。"""
    result = []
    for url in urls:
        feat = extract_features(url)
        feat_list = [feat.get(name, 0) for name in FEATURE_NAMES]
        result.append(feat_list)
    return pd.DataFrame(result, columns=FEATURE_NAMES)
