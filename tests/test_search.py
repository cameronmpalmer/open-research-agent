"""Tests for the search provider dispatch (Decodo primary, Firecrawl fallback)."""

import json

import ora.tools.search as search_mod
from ora.config import ORASettings, SearchSettings
from ora.tools.search import web_search


def _settings(provider="decodo", fallback=True):
    s = ORASettings()
    s.search = SearchSettings(provider=provider, fallback_to_firecrawl=fallback)
    return s


class FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


DECODO_OK = {
    "results": [
        {
            "content": {
                "results": {
                    "organic": [
                        {"title": "Paper A", "url": "https://a.example/1", "description": "alpha"},
                        {"title": "Paper B", "url": "https://b.example/2", "description": "beta"},
                    ]
                }
            }
        }
    ]
}


def test_decodo_results_are_formatted(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings())
    monkeypatch.setenv("DECODO_USERNAME", "u")
    monkeypatch.setenv("DECODO_PASSWORD", "p")
    calls = {}

    def fake_post(url, **kw):
        calls["url"] = url
        calls["body"] = json.loads(kw["data"])
        return FakeResp(DECODO_OK)

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://a.example/1" in out
    assert "Paper A" in out
    assert calls["url"].endswith("/v2/scrape")
    assert calls["body"]["target"] == "google_search"


def test_decodo_missing_credentials_falls_back_to_firecrawl(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings())
    monkeypatch.delenv("DECODO_USERNAME", raising=False)
    monkeypatch.delenv("DECODO_PASSWORD", raising=False)

    def fake_post(url, **kw):
        assert "/v1/search" in url
        return FakeResp(
            {"success": True, "data": [{"title": "FB", "url": "https://f.example", "description": "x"}]}
        )

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://f.example" in out


def test_decodo_api_failure_falls_back(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings())
    monkeypatch.setenv("DECODO_USERNAME", "u")
    monkeypatch.setenv("DECODO_PASSWORD", "p")
    seen = []

    def fake_post(url, **kw):
        seen.append(url)
        if "/v2/scrape" in url:
            return FakeResp({"status": "failed", "status_code": 613, "message": "nope"})
        return FakeResp(
            {"success": True, "data": [{"title": "FB", "url": "https://f.example", "description": "x"}]}
        )

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://f.example" in out
    assert any("/v2/scrape" in u for u in seen) and any("/v1/search" in u for u in seen)


def test_no_results_returns_explicit_message(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=False))
    monkeypatch.setenv("DECODO_USERNAME", "u")
    monkeypatch.setenv("DECODO_PASSWORD", "p")
    monkeypatch.setattr(
        search_mod.requests, "post", lambda url, **kw: FakeResp({"results": []})
    )
    out = search_mod._search("q", 5)
    assert "No search results found" in out


def test_decodo_empty_results_do_not_trigger_fallback(monkeypatch):
    """Decodo succeeding with zero results is an answer, not a failure.

    Fallback must NOT run even when fallback_to_firecrawl is enabled.
    """
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_USERNAME", "u")
    monkeypatch.setenv("DECODO_PASSWORD", "p")
    called = []

    def fake_post(url, **kw):
        called.append(url)
        return FakeResp({"results": []})

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "No search results found" in out
    assert all("/v1/search" not in u for u in called), f"fallback fired: {called}"


def test_limit_is_respected(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings())
    monkeypatch.setenv("DECODO_USERNAME", "u")
    monkeypatch.setenv("DECODO_PASSWORD", "p")
    payload = {
        "results": [
            {
                "content": {
                    "results": {
                        "organic": [
                            {"title": "A", "url": "https://a.example", "description": "a"},
                            {"title": "B", "url": "https://b.example", "description": "b"},
                            {"title": "C", "url": "https://c.example", "description": "c"},
                        ]
                    }
                }
            }
        ]
    }
    monkeypatch.setattr(search_mod.requests, "post", lambda url, **kw: FakeResp(payload))
    out = search_mod._search("q", 2)
    assert out.count("https://") == 2
    assert "https://c.example" not in out


def test_malformed_payload_returns_error_not_raise(monkeypatch):
    monkeypatch.setenv("DECODO_USERNAME", "u")
    monkeypatch.setenv("DECODO_PASSWORD", "p")

    # Decodo path.
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=False))
    for payload in (None, []):
        monkeypatch.setattr(
            search_mod.requests, "post", lambda url, _payload=payload, **kw: FakeResp(_payload)
        )
        out = search_mod._search("q", 5)
        assert out.startswith("Search error:"), f"decodo payload={payload!r} -> {out!r}"

    # Firecrawl path must harden against non-dict bodies too.
    monkeypatch.setattr(
        search_mod, "load_config", lambda: _settings(provider="firecrawl", fallback=False)
    )
    for payload in (None, []):
        monkeypatch.setattr(
            search_mod.requests, "post", lambda url, _payload=payload, **kw: FakeResp(_payload)
        )
        out = search_mod._search("q", 5)
        assert out.startswith("Search error:"), f"firecrawl payload={payload!r} -> {out!r}"


FIRECRAWL_OK = {
    "success": True,
    "data": [{"title": "FB", "url": "https://f.example", "description": "x"}],
}


def test_decodo_missing_results_key_falls_back(monkeypatch):
    """A 200 dict body without a `results` key is a failure, not an empty answer."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_USERNAME", "u")
    monkeypatch.setenv("DECODO_PASSWORD", "p")
    seen = []

    def fake_post(url, **kw):
        seen.append(url)
        if "/v2/scrape" in url:
            return FakeResp({"detail": "not a search response"})
        return FakeResp(FIRECRAWL_OK)

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://f.example" in out
    assert any("/v1/search" in u for u in seen)


def test_decodo_http_error_falls_back(monkeypatch):
    """HTTP failures (401/429/5xx) must trigger fallback, not read as empty.

    The error body deliberately carries an empty `results` list: without the
    status-code check it would be misread as a legitimate zero-result success.
    """
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_USERNAME", "u")
    monkeypatch.setenv("DECODO_PASSWORD", "p")
    seen = []
    status = {"code": 401}

    def fake_post(url, **kw):
        seen.append(url)
        if "/v2/scrape" in url:
            return FakeResp({"results": []}, status=status["code"])
        return FakeResp(FIRECRAWL_OK)

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    for code in (401, 429, 500):
        status["code"] = code
        seen.clear()
        out = search_mod._search("q", 5)
        assert "https://f.example" in out, f"status={code} -> {out!r}"
        assert any("/v1/search" in u for u in seen), f"status={code} no fallback"


def test_firecrawl_provider_happy_path(monkeypatch):
    monkeypatch.setattr(
        search_mod, "load_config", lambda: _settings(provider="firecrawl", fallback=True)
    )
    seen = []

    def fake_post(url, **kw):
        seen.append(url)
        return FakeResp(FIRECRAWL_OK)

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://f.example" in out
    assert all("/v2/scrape" not in u for u in seen), f"decodo contacted: {seen}"


def test_firecrawl_api_url_env_override(monkeypatch):
    monkeypatch.setattr(
        search_mod, "load_config", lambda: _settings(provider="firecrawl", fallback=True)
    )
    monkeypatch.setenv("FIRECRAWL_API_URL", "http://selfhosted.example:3002")
    seen = []

    def fake_post(url, **kw):
        seen.append(url)
        return FakeResp(FIRECRAWL_OK)

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    search_mod._search("q", 5)
    assert seen == ["http://selfhosted.example:3002/v1/search"]


class TestSearchTool:
    def test_tool_has_name(self):
        assert web_search.name == "web_search"

    def test_tool_has_description(self):
        assert "search" in web_search.description.lower()
