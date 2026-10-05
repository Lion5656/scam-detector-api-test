"""呼叫 URL 模型前的網域檢查：台灣受管制後綴白名單、短網址與可疑 tw 字樣判斷。"""

import re
from dataclasses import dataclass
from urllib.parse import urlparse

import tldextract

from backend.utils.features import COMMON_DOMAINS

REGULATED_SUFFIXES = frozenset({"com.tw", "org.tw", "net.tw", "idv.tw", "gov.tw", "edu.tw", "mil.tw"})

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
    is_shortened: bool
    has_suspicious_tw: bool


def get_suffix_signals(url: str) -> SuffixSignals:
    """解析 URL 的公開後綴，回傳白名單、短網址與可疑 tw 旗標。

    is_regulated：後綴屬於 REGULATED_SUFFIXES，例如 moe.gov.tw。
    is_shortened：註冊網域屬於 SHORTENER_DOMAINS，例如 reurl.cc/abc、bit.ly/xyz。
    has_suspicious_tw：註冊網域不在 COMMON_DOMAINS，且符合以下任一：
        - 後綴是 tw 但不在白名單，例如 ezpay.tw、abc.game.tw
        - tw 黏在網域主體結尾，例如 ezpaytw.com
        - tw 出現在子網域或網域主體，例如 gov.tw.login-check.xyz、tw-post.top
    """
    # 必須用 hostname 而非 netloc：https://x.gov.tw@evil.xyz 的 netloc 含帳密段，會被誤判成 gov.tw。
    host = (urlparse(url).hostname or "").rstrip(".")
    ext = tldextract.extract(host)
    suffix = ext.suffix
    is_regulated = suffix in REGULATED_SUFFIXES

    is_unregulated_tw = not is_regulated and (suffix == "tw" or suffix.endswith(".tw"))

    # endswith 同時涵蓋獨立的 tw 與黏在結尾的 govtw；不比對 tw 開頭，否則 twitter、twitch 都會命中。
    host_part = f"{ext.subdomain}.{ext.domain}" if ext.subdomain else ext.domain
    tokens = [t for t in HOST_TOKEN_SPLIT_RE.split(host_part) if t]
    has_tw_outside_suffix = any(t.endswith("tw") for t in tokens)

    root_domain = f"{ext.domain}.{suffix}"
    is_common_domain = root_domain in COMMON_DOMAINS

    return SuffixSignals(
        suffix=suffix,
        is_regulated=is_regulated,
        is_shortened=root_domain in SHORTENER_DOMAINS,
        has_suspicious_tw=(is_unregulated_tw or has_tw_outside_suffix) and not is_common_domain,
    )
