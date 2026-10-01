"""Tests for the search provider dispatch (Decodo primary, Firecrawl fallback)."""

import json

import ora.tools.search as search_mod
from ora.config import ORASettings, SearchSettings
from ora.tools.search import web_search


def _settings(provider="decodo", fallback=True):
    s = ORASettings()
    s.search = SearchSettings(provider=provider, fallback_to_firecrawl=fallback)
    return s


class _FakeTool:
    """Minimal stand-in for a LangChain @tool object."""

    def __init__(self, fn):
        self._fn = fn

    def invoke(self, payload):
        return self._fn(payload)


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


class TestSearchTool:
    def test_tool_has_name(self):
        assert web_search.name == "web_search"

    def test_tool_has_description(self):
        assert "search" in web_search.description.lower()
