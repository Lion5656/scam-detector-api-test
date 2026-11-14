"""呼叫 URL 模型前的網域檢查：台灣受管制後綴、短網址與可疑 tw 字樣判斷。"""

import re
from dataclasses import dataclass
from urllib.parse import urlparse

import tldextract

from backend.utils.features import COMMON_DOMAINS

# 僅限政府機關、學校、軍方註冊，不必再交給模型判斷
RESTRICTED_SUFFIXES = frozenset({"gov.tw", "edu.tw", "mil.tw"})
# 需實名（公司、組織、個人）註冊，但仍可能遭入侵或被冒名申請，仍需經過模型
REGISTERED_SUFFIXES = frozenset({"com.tw", "org.tw", "net.tw", "idv.tw"})
REGULATED_SUFFIXES = RESTRICTED_SUFFIXES | REGISTERED_SUFFIXES

# 只收可導向任意網站的公用短網址；youtu.be、amzn.to、lin.ee 這類只導回自家品牌的不列入
SHORTENER_DOMAINS = frozenset({
    # 台灣常見大眾短網址
    "reurl.cc", "ppt.cc", "pse.is", "lihi.cc", "lihi1.com", "lihi1.cc", "lihi2.com", "lihi3.com",
    # 國際
    "bit.ly", "j.mp", "tinyurl.com", "t.ly", "t.co", "goo.gl", "is.gd", "v.gd", "ow.ly", "buff.ly",
    "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc", "bit.do", "s.id", "rebrand.ly", "gg.gg",
})

HOST_TOKEN_SPLIT_RE = re.compile(r"[.\-_\d]+")


@dataclass(frozen=True)
class SuffixSignals:
    suffix: str
    is_regulated: bool
    is_restricted: bool
    is_shortened: bool
    has_unregulated_tw_suffix: bool
    has_tw_in_host: bool

    @property
    def has_suspicious_tw(self) -> bool:
        return self.has_unregulated_tw_suffix or self.has_tw_in_host


def get_suffix_signals(url: str) -> SuffixSignals:
    """解析 URL 的公開後綴，回傳管制後綴、短網址與可疑 tw 旗標。

    is_regulated：後綴屬於 REGULATED_SUFFIXES，例如 moe.gov.tw、example.com.tw。
    is_restricted：後綴屬於 RESTRICTED_SUFFIXES（gov.tw／edu.tw／mil.tw）。
    is_shortened：註冊網域屬於 SHORTENER_DOMAINS，例如 reurl.cc/abc、bit.ly/xyz。
    兩個 tw 旗標都排除 COMMON_DOMAINS：
        has_unregulated_tw_suffix：後綴是 tw 但不在管制清單，例如 ezpay.tw、abc.game.tw
        has_tw_in_host：tw 出現在子網域或網域主體，例如 ezpaytw.com、gov.tw.login-check.xyz
    """
    # 必須用 hostname 而非 netloc：https://x.gov.tw@evil.xyz 的 netloc 含帳密段，會被誤判成 gov.tw。
    host = (urlparse(url).hostname or "").rstrip(".")
    ext = tldextract.extract(host)
    suffix = ext.suffix
    is_regulated = suffix in REGULATED_SUFFIXES
    root_domain = f"{ext.domain}.{suffix}"
    is_common_domain = root_domain in COMMON_DOMAINS

    is_unregulated_tw = not is_regulated and (suffix == "tw" or suffix.endswith(".tw"))

    # endswith 同時涵蓋獨立的 tw 與黏在結尾的 govtw；不比對 tw 開頭，否則 twitter、twitch 都會命中。
    host_part = f"{ext.subdomain}.{ext.domain}" if ext.subdomain else ext.domain
    tokens = [t for t in HOST_TOKEN_SPLIT_RE.split(host_part) if t]
    has_tw_outside_suffix = any(t.endswith("tw") for t in tokens)

    return SuffixSignals(
        suffix=suffix,
        is_regulated=is_regulated,
        is_restricted=suffix in RESTRICTED_SUFFIXES,
        is_shortened=root_domain in SHORTENER_DOMAINS,
        has_unregulated_tw_suffix=is_unregulated_tw and not is_common_domain,
        has_tw_in_host=has_tw_outside_suffix and not is_common_domain,
    )
