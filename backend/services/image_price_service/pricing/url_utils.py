"""市場價格來源 URL 的共用正規化工具。"""

from urllib.parse import urlsplit, urlunsplit


def normalize_url(url: str) -> str:
    """正規化 HTTP(S) 來源 URL，供來源網址驗證使用。"""
    try:
        parsed = urlsplit(url)
    except ValueError:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") or "/",
            parsed.query,
            "",
        )
    )


__all__ = ["normalize_url"]
