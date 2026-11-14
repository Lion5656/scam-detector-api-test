import httpx
import pytest

from backend.services.url_service import redirect_resolver
from backend.services.url_service.redirect_resolver import resolve_final_url


@pytest.fixture
def public_dns(monkeypatch):
    """測試主機一律視為公網，避免真的查 DNS；私有位址測試另外以真實解析驗證。"""
    monkeypatch.setattr(redirect_resolver, "_is_public_host", lambda host, port: True)


def make_transport(routes):
    def handler(request):
        location = routes.get(str(request.url))
        if location is None:
            return httpx.Response(200, text="final page")
        return httpx.Response(302, headers={"location": location})

    return httpx.MockTransport(handler)


def test_follows_redirect_chain_to_final_url(public_dns):
    transport = make_transport({
        "https://reurl.cc/abc": "https://bit.ly/next",
        "https://bit.ly/next": "/landing?x=1",
    })

    result = resolve_final_url("https://reurl.cc/abc", transport=transport)

    assert result == "https://bit.ly/landing?x=1"


def test_no_redirect_returns_none(public_dns):
    assert resolve_final_url("https://reurl.cc/preview", transport=make_transport({})) is None


def test_too_many_redirects_returns_none(public_dns):
    transport = make_transport({f"https://a.com/{i}": f"https://a.com/{i + 1}" for i in range(10)})

    assert resolve_final_url("https://a.com/0", max_redirects=3, transport=transport) is None


def test_network_error_returns_none(public_dns):
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    assert resolve_final_url("https://reurl.cc/abc", transport=httpx.MockTransport(handler)) is None


@pytest.mark.parametrize("target", [
    "http://127.0.0.1/admin",
    "http://10.0.0.5/",
    "http://169.254.169.254/latest/meta-data/",
    "http://[::1]/",
    "http://localhost/",
])
def test_redirect_to_private_address_is_blocked(target):
    requested = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": target})

    original_check = redirect_resolver._is_public_host

    def only_first_hop_public(host, port):
        return host == "reurl.cc" or original_check(host, port)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(redirect_resolver, "_is_public_host", only_first_hop_public)
        result = resolve_final_url("https://reurl.cc/abc", transport=httpx.MockTransport(handler))

    assert result is None
    assert requested == ["https://reurl.cc/abc"]


@pytest.mark.parametrize(
    "target", ["ftp://example.com/file", "https://example.com:8443/", "file:///etc/passwd"]
)
def test_disallowed_scheme_or_port_is_blocked(public_dns, target):
    transport = make_transport({"https://reurl.cc/abc": target})

    assert resolve_final_url("https://reurl.cc/abc", transport=transport) is None
