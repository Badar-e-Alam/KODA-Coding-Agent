"""The request guard in koda/tools/web.py refuses internal addresses (SSRF).

No real network: DNS (``socket.getaddrinfo``) and HTTP (``httpx.get``) are mocked.

Scope note: ``read_webpage`` hands the URL to r.jina.ai, so redirects are
followed on Jina's side, where this guard cannot see them. The redirect tests
prove that KODA itself never follows a 3xx and never sends a request to the
internal address; they do not cover what Jina does.
"""

from __future__ import annotations

import socket

import httpx
import pytest

from koda.tools import web

PUBLIC_IP = "93.184.216.34"


@pytest.fixture
def dns(monkeypatch):
    """Map hostnames to fake addresses; literal IPs resolve to themselves."""
    table: dict[str, list[str]] = {"public.example": [PUBLIC_IP]}

    def _getaddrinfo(host, port, *args, **kwargs):
        ips = table.get(host, [host])
        out = []
        for ip in ips:
            if ":" in ip:
                out.append((socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, 0, 0, 0)))
            else:
                out.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)))
        return out

    monkeypatch.setattr(web.socket, "getaddrinfo", _getaddrinfo)
    return table


@pytest.fixture
def http(monkeypatch):
    """Record httpx.get calls; reply 200 unless a redirect target is set."""
    state: dict = {"calls": [], "redirect_to": None}

    def _get(url, **kwargs):
        state["calls"].append((url, kwargs))
        request = httpx.Request("GET", url)
        if state["redirect_to"]:
            return httpx.Response(302, headers={"location": state["redirect_to"]},
                                  request=request)
        return httpx.Response(200, text="# page", request=request)

    monkeypatch.setattr(web.httpx, "get", _get)
    return state


def _assert_refused(url, http):
    out = web.read_webpage.invoke({"url": url})
    assert out.startswith("Error: refusing to fetch private/internal address"), out
    assert http["calls"] == []


# ── one test per blocked address class ─────────────────────────────────


def test_localhost_name_refused(dns, http):
    dns["localhost"] = ["127.0.0.1"]
    _assert_refused("http://localhost:8080/admin", http)


def test_loopback_127_8_refused(dns, http):
    _assert_refused("http://127.5.6.7/", http)


def test_private_10_8_refused(dns, http):
    _assert_refused("http://10.1.2.3/", http)


def test_private_172_16_12_refused(dns, http):
    _assert_refused("http://172.16.0.1/", http)
    _assert_refused("http://172.31.255.254/", http)


def test_private_192_168_16_refused(dns, http):
    _assert_refused("http://192.168.1.1/", http)


def test_cloud_metadata_169_254_169_254_refused(dns, http):
    _assert_refused("http://169.254.169.254/latest/meta-data/", http)


def test_ipv6_loopback_refused(dns, http):
    _assert_refused("http://[::1]:8000/", http)


def test_hostname_resolving_to_internal_refused(dns, http):
    dns["intranet.example"] = ["10.0.0.5"]
    _assert_refused("https://intranet.example/", http)


def test_hostname_with_any_internal_address_refused(dns, http):
    dns["mixed.example"] = [PUBLIC_IP, "192.168.0.10"]
    _assert_refused("https://mixed.example/", http)


def test_non_http_scheme_refused(dns, http):
    out = web.read_webpage.invoke({"url": "file:///etc/passwd"})
    assert out.startswith("Error: only http/https")
    assert http["calls"] == []


def test_public_address_allowed(dns, http):
    out = web.read_webpage.invoke({"url": "https://public.example/docs"})
    assert out == "# page"
    assert [c[0] for c in http["calls"]] == ["https://r.jina.ai/https://public.example/docs"]


# ── redirects to internal addresses ────────────────────────────────────


@pytest.mark.parametrize(
    "target",
    [
        "http://localhost/",
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
    ],
)
def test_redirect_to_internal_address_not_followed(dns, http, target):
    http["redirect_to"] = target
    with pytest.raises(httpx.HTTPStatusError):
        web.read_webpage.invoke({"url": "https://public.example/"})
    # Exactly one request, to the reader service; the 3xx is never followed.
    assert len(http["calls"]) == 1
    url, kwargs = http["calls"][0]
    assert url.startswith("https://r.jina.ai/")
    assert kwargs.get("follow_redirects") is not True
