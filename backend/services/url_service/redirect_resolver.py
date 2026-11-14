"""追蹤短網址的 HTTP 轉址，找出實際導向的網址。

這裡會依使用者輸入對外發出請求，因此每一跳都先檢查目標主機只解析到公網 IP，
並限制協定、埠號、轉址次數與總時間，避免被用來探測內網（SSRF）。
已知限制：檢查 DNS 與實際連線之間仍有 DNS rebinding 的時間差。
"""

import ipaddress
import logging
import socket
import time
from urllib.parse import urljoin, urlsplit

import httpx

from backend.core.config import settings

logger = logging.getLogger(__name__)

_ALLOWED_PORTS = {"http": (None, 80), "https": (None, 443)}
_REDIRECT_STATUS = {301, 302, 303, 307, 308}
# 部分短網址服務會拒絕沒有 User-Agent 的請求；只用通用名稱，不帶任何使用者資訊
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; url-risk-checker/1.0)"}


def _is_public_host(host: str, port: int) -> bool:
    """主機的所有解析結果都必須是公網 IP；解析失敗視為不安全。"""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global:
            return False
    return bool(infos)


def _is_allowed_target(url: str) -> bool:
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        return False
    if parts.scheme not in _ALLOWED_PORTS or port not in _ALLOWED_PORTS[parts.scheme]:
        return False
    if not parts.hostname:
        return False
    return _is_public_host(parts.hostname, port or (443 if parts.scheme == "https" else 80))


def resolve_final_url(
    url: str,
    *,
    timeout: float | None = None,
    max_redirects: int | None = None,
    transport: httpx.BaseTransport | None = None,
) -> str | None:
    """回傳轉址鏈的最終網址；沒有轉址、超過次數、逾時或目標不安全時回傳 None。"""
    timeout = settings.URL_EXPAND_TIMEOUT_SECONDS if timeout is None else timeout
    max_redirects = settings.URL_EXPAND_MAX_REDIRECTS if max_redirects is None else max_redirects
    deadline = time.monotonic() + timeout
    current = url

    with httpx.Client(follow_redirects=False, headers=_HEADERS, transport=transport) as client:
        for hop in range(max_redirects + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.info("短網址展開逾時 url=%s", url)
                return None
            if not _is_allowed_target(current):
                logger.warning("短網址導向不允許的目標 url=%s target=%s", url, current)
                return None

            # 只需要回應標頭：用串流 GET 取得狀態碼與 Location 後立即關閉，不下載內容。
            # 不用 HEAD，因為部分短網址服務對 HEAD 回 405 或不回傳轉址
            try:
                with client.stream("GET", current, timeout=remaining) as response:
                    location = response.headers.get("location")
                    status = response.status_code
            except httpx.HTTPError as exc:
                logger.info("短網址展開失敗 url=%s error=%s", url, exc.__class__.__name__)
                return None

            if status not in _REDIRECT_STATUS or not location:
                # hop 為 0 代表短網址本身沒有轉址（例如預覽頁），無法得知目標
                return current if hop > 0 else None
            current = urljoin(current, location)

    logger.info("短網址轉址次數超過上限 url=%s", url)
    return None
