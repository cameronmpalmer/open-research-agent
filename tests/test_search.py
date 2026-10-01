"""Tests for the search provider dispatch (Decodo primary, Firecrawl fallback)."""

import json

import pytest

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


DECODO_LIVE = {
    "results": [
        {
            "content": {
                "results": {
                    "results": {
                        "organic": [
                            {
                                "title": "Live A",
                                "url": "https://live.example/1",
                                "desc": "live alpha",
                            }
                        ]
                    },
                    "errors": [],
                }
            }
        }
    ]
}


def test_decodo_live_two_level_nesting_uses_desc(monkeypatch):
    """Live 2026-10-01 shape: content.results.results.organic with a `desc` field.

    The single-level fixture above does not match the live Decodo response; this
    guards the deeper nesting and the `desc` description key actually returned.
    """
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings())
    monkeypatch.setenv("DECODO_USERNAME", "u")
    monkeypatch.setenv("DECODO_PASSWORD", "p")
    monkeypatch.setattr(search_mod.requests, "post", lambda url, **kw: FakeResp(DECODO_LIVE))
    out = search_mod._search("q", 5)
    assert "https://live.example/1" in out
    assert "Live A" in out
    assert "live alpha" in out


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


def test_decodo_content_error_falls_back(monkeypatch):
    """A 200 body carrying content.errors must trigger the fallback.

    Live Decodo (verified 2026-10-01) nests its own error list at
    content.errors; an empty list means success, a populated one is a failure.
    """
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_USERNAME", "u")
    monkeypatch.setenv("DECODO_PASSWORD", "p")
    seen = []

    def fake_post(url, **kw):
        seen.append(url)
        if "/v2/scrape" in url:
            return FakeResp(
                {"results": [{"content": {"errors": ["quota exceeded"], "results": {}}}]}
            )
        return FakeResp(FIRECRAWL_OK)

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://f.example" in out
    assert any("/v1/search" in u for u in seen)


def test_decodo_nested_results_error_falls_back(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_USERNAME", "u")
    monkeypatch.setenv("DECODO_PASSWORD", "p")
    seen = []

    def fake_post(url, **kw):
        seen.append(url)
        if "/v2/scrape" in url:
            return FakeResp(
                {"results": [{"content": {"results": {"errors": ["bad parse"]}}}]}
            )
        return FakeResp(FIRECRAWL_OK)

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://f.example" in out
    assert any("/v1/search" in u for u in seen)


def test_decodo_bad_content_status_falls_back(monkeypatch):
    """content.status_code is Decodo's own parse status; 12000 (and 200) mean
    success, anything else present is a failure and must fall back."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_USERNAME", "u")
    monkeypatch.setenv("DECODO_PASSWORD", "p")
    seen = []

    def fake_post(url, **kw):
        seen.append(url)
        if "/v2/scrape" in url:
            return FakeResp(
                {"results": [{"content": {"status_code": 10000, "results": {}}}]}
            )
        return FakeResp(FIRECRAWL_OK)

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://f.example" in out
    assert any("/v1/search" in u for u in seen)


def test_live_success_content_status_is_not_treated_as_error(monkeypatch):
    """content.status_code == 12000 is a live success; it must NOT be a failure."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=False))
    monkeypatch.setenv("DECODO_USERNAME", "u")
    monkeypatch.setenv("DECODO_PASSWORD", "p")

    def fake_post(url, **kw):
        return FakeResp(
            {
                "results": [
                    {
                        "content": {
                            "status_code": 12000,
                            "errors": [],
                            "results": {
                                "results": {
                                    "organic": [
                                        {"title": "T", "url": "https://x.example", "desc": "d"}
                                    ]
                                }
                            },
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://x.example" in out


def test_unknown_provider_warns_and_uses_firecrawl(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(provider="decodo "))
    monkeypatch.setenv("FIRECRAWL_API_URL", "http://selfhosted.example:3002")

    def fake_post(url, **kw):
        return FakeResp(FIRECRAWL_OK)

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    with pytest.warns(UserWarning, match="Unknown search provider"):
        out = search_mod._search("q", 5)
    assert "https://f.example" in out


def test_firecrawl_base_url_trailing_slash_is_stripped(monkeypatch):
    monkeypatch.setattr(
        search_mod, "load_config", lambda: _settings(provider="firecrawl", fallback=True)
    )
    monkeypatch.setenv("FIRECRAWL_API_URL", "http://selfhosted.example:3002/")
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
