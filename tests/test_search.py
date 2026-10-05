"""Tests for the search provider dispatch (Decodo primary, Firecrawl fallback).

Decodo fixtures follow the documented Web Scraping API response contract:
HTTP 200 success, 204 job-not-complete, ``>= 400`` failure; body
``status == "failed"`` or a top-level ``status_code`` outside ``{200, 202}``;
per-task ``parse_status_code`` usable in ``{12000, 12004, 12005}`` with a
``content.status_code`` fallback in ``{200, 12000}``; organic results at
``content.results.results.organic`` with ``desc`` snippets.
"""

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


FIRECRAWL_OK = {
    "success": True,
    "data": [{"title": "FB", "url": "https://f.example", "description": "x"}],
}

LIVE_ORGANIC = [
    {"title": "Live A", "url": "https://live.example/1", "desc": "live alpha"},
    {"title": "Live B", "url": "https://live.example/2", "desc": "live beta"},
]


def _live_content(organic, *, parse_status=12000, errors=None, content_status=None):
    """A documented google_search ``content`` object."""
    results: dict = {"results": {"organic": organic}}
    if parse_status is not None:
        results["parse_status_code"] = parse_status
    content: dict = {"results": results}
    if errors is not None:
        content["errors"] = errors
    if content_status is not None:
        content["status_code"] = content_status
    return content


def _live_payload(organic, **kwargs):
    return {"results": [{"content": _live_content(organic, **kwargs)}]}


def _decodo_then_firecrawl(payload, status=200):
    """A fake_post serving ``payload`` for Decodo and FIRECRAWL_OK for Firecrawl."""
    seen: list[str] = []

    def fake_post(url, **kw):
        seen.append(url)
        if "/v2/scrape" in url:
            return FakeResp(payload, status=status)
        return FakeResp(FIRECRAWL_OK)

    return fake_post, seen


def test_decodo_request_targets_google_search(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings())
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    calls = {}

    def fake_post(url, **kw):
        calls["url"] = url
        calls["body"] = json.loads(kw["data"])
        return FakeResp(_live_payload(LIVE_ORGANIC))

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    search_mod._search("q", 5)
    assert calls["url"].endswith("/v2/scrape")
    assert calls["body"]["target"] == "google_search"


def test_decodo_live_success_returns_results_no_fallback(monkeypatch):
    """Documented success: parsed results win and the fallback never runs."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    fake_post, seen = _decodo_then_firecrawl(_live_payload(LIVE_ORGANIC, errors=[]))
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://live.example/1" in out
    assert "Live A" in out
    assert "live alpha" in out
    assert all("/v1/search" not in u for u in seen), f"fallback fired: {seen}"


def test_decodo_item_prefers_desc_over_description(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    organic = [
        {"title": "T", "url": "https://x.example", "desc": "from-desc", "description": "fallback"}
    ]
    monkeypatch.setattr(
        search_mod.requests, "post", lambda url, **kw: FakeResp(_live_payload(organic))
    )
    out = search_mod._search("q", 5)
    assert "from-desc" in out
    assert "fallback" not in out


def test_decodo_single_level_nesting(monkeypatch):
    """Tolerated older shape: content.results.organic (one level shallower)."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings())
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    payload = {
        "results": [
            {
                "content": {
                    "results": {
                        "parse_status_code": 12000,
                        "organic": [{"title": "One", "url": "https://one.example", "desc": "one"}],
                    }
                }
            }
        ]
    }
    monkeypatch.setattr(search_mod.requests, "post", lambda url, **kw: FakeResp(payload))
    out = search_mod._search("q", 5)
    assert "https://one.example" in out


def test_decodo_organic_on_content(monkeypatch):
    """Tolerated shape: organic directly on content."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings())
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    payload = {
        "results": [
            {
                "content": {
                    "parse_status_code": 12000,
                    "organic": [{"title": "Direct", "url": "https://direct.example", "desc": "d"}],
                }
            }
        ]
    }
    monkeypatch.setattr(search_mod.requests, "post", lambda url, **kw: FakeResp(payload))
    out = search_mod._search("q", 5)
    assert "https://direct.example" in out


def test_decodo_live_empty_organic_is_empty_no_fallback(monkeypatch):
    """A usable parse with an empty organic list is a legitimate empty answer."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    fake_post, seen = _decodo_then_firecrawl(_live_payload([], errors=[]))
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "No search results found" in out
    assert all("/v1/search" not in u for u in seen), f"fallback fired: {seen}"


def test_empty_results_list_is_empty_no_fallback(monkeypatch):
    """No task entries, no failure signal: a legitimate empty parse."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    fake_post, seen = _decodo_then_firecrawl({"results": []})
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "No search results found" in out
    assert all("/v1/search" not in u for u in seen), f"fallback fired: {seen}"


def test_decodo_613_no_results_envelope_falls_back(monkeypatch):
    """The documented no-results state is a failure, not an empty success."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    payload = {
        "status": "failed",
        "status_code": 613,
        "message": "We were not able to scrape the target",
        "task_id": "t",
    }
    fake_post, seen = _decodo_then_firecrawl(payload)
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://f.example" in out
    assert any("/v1/search" in u for u in seen)


def test_decodo_613_no_results_envelope_fallback_disabled(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=False))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    called: list[str] = []

    def fake_post(url, **kw):
        called.append(url)
        return FakeResp({"status": "failed", "status_code": 613, "message": "not able to scrape"})

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert out.startswith("Search error:") and "613" in out, out
    assert all("/v1/search" not in u for u in called), f"fallback fired: {called}"


def test_decodo_204_then_200_retries(monkeypatch):
    """HTTP 204 means the job is not complete; retry briefly, then succeed."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    responses = iter([FakeResp({}, status=204), FakeResp(_live_payload(LIVE_ORGANIC), status=200)])
    counts = {"post": 0, "sleep": 0}

    def fake_post(url, **kw):
        counts["post"] += 1
        return next(responses)

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    monkeypatch.setattr(
        search_mod.time, "sleep", lambda seconds: counts.__setitem__("sleep", counts["sleep"] + 1)
    )
    out = search_mod._search("q", 5)
    assert "https://live.example/1" in out
    assert counts["post"] == 2
    assert counts["sleep"] == 1


def test_decodo_204_exhausted_falls_back(monkeypatch):
    """When every call is 204, give up after the bounded retries and fall back."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    decodo_calls: list[str] = []
    sleeps: list[float] = []

    def fake_post(url, **kw):
        decodo_calls.append(url)
        if "/v2/scrape" in url:
            return FakeResp({}, status=204)
        return FakeResp(FIRECRAWL_OK)

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    monkeypatch.setattr(search_mod.time, "sleep", lambda seconds: sleeps.append(seconds))
    out = search_mod._search("q", 5)
    assert "https://f.example" in out
    assert len(decodo_calls) == 4  # 3 Decodo 204s + 1 Firecrawl fallback
    assert any("/v1/search" in u for u in decodo_calls)
    assert len(sleeps) == 2  # sleeps only between the bounded retries


def test_decodo_401_top_level_error_falls_back(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    payload = {
        "status": "failed",
        "status_code": 401,
        "message": "Incorrect username or password",
    }
    fake_post, seen = _decodo_then_firecrawl(payload, status=401)
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://f.example" in out
    assert any("/v1/search" in u for u in seen)


def test_decodo_400_error_message_is_surfaced(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=False))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    payload = {"status": "failed", "message": "The target is invalid"}

    def fake_post(url, **kw):
        return FakeResp(payload, status=400)

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert out.startswith("Search error: decodo HTTP 400")
    assert "The target is invalid" in out


def test_decodo_top_level_status_code_failure_falls_back(monkeypatch):
    """A body status_code outside {200, 202} is a failure even at HTTP 200."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    payload = {"status_code": 400, "message": "bad", "results": [{"content": _live_content([])}]}
    fake_post, seen = _decodo_then_firecrawl(payload)
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://f.example" in out
    assert any("/v1/search" in u for u in seen)


def test_decodo_top_level_status_code_202_is_success(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    payload = {"status_code": 202, "results": [{"content": _live_content(LIVE_ORGANIC)}]}
    fake_post, seen = _decodo_then_firecrawl(payload)
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://live.example/1" in out
    assert all("/v1/search" not in u for u in seen), f"fallback fired: {seen}"


@pytest.mark.parametrize("code", [12002, 12003, 12006, 12007, 12008, 12009])
def test_decodo_parse_status_failure_codes_fall_back(monkeypatch, code):
    """Documented parse-failure codes must fall back rather than read as empty."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    payload = _live_payload([], parse_status=code)
    fake_post, seen = _decodo_then_firecrawl(payload)
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://f.example" in out, f"code={code} -> {out!r}"
    assert any("/v1/search" in u for u in seen), f"code={code} no fallback"


@pytest.mark.parametrize("code", [12004, 12005])
def test_decodo_parse_status_partial_codes_are_usable(monkeypatch, code):
    """Partial-success parse codes (12004/12005) are usable, not failures."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    payload = _live_payload(LIVE_ORGANIC, parse_status=code, errors=[])
    fake_post, seen = _decodo_then_firecrawl(payload)
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://live.example/1" in out, f"code={code} -> {out!r}"
    assert all("/v1/search" not in u for u in seen), f"code={code} fallback fired"


def test_decodo_html_content_is_failure(monkeypatch):
    """Observed live: a missing Authorization header returns content as HTML."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    payload = {"results": [{"content": "<html><body>Sign in</body></html>"}]}
    fake_post, seen = _decodo_then_firecrawl(payload)
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://f.example" in out
    assert any("/v1/search" in u for u in seen)


def test_decodo_content_errors_non_empty_falls_back(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    payload = _live_payload([], errors=["quota exceeded"])
    fake_post, seen = _decodo_then_firecrawl(payload)
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://f.example" in out
    assert any("/v1/search" in u for u in seen)


def test_nested_warning_errors_with_organic_keep_results(monkeypatch):
    """Warnings beside valid organic results must not discard the results."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    payload = _live_payload(LIVE_ORGANIC, errors=["warn"])
    fake_post, seen = _decodo_then_firecrawl(payload)
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://live.example/1" in out
    assert all("/v1/search" not in u for u in seen), f"fallback fired: {seen}"


def test_decodo_bad_content_status_when_parse_absent_falls_back(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    payload = {
        "results": [
            {
                "content": {
                    "status_code": 500,
                    "results": {"results": {"organic": []}},
                }
            }
        ]
    }
    fake_post, seen = _decodo_then_firecrawl(payload)
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://f.example" in out
    assert any("/v1/search" in u for u in seen)


def test_decodo_content_status_200_without_parse_code_is_usable(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    payload = {
        "results": [
            {
                "content": {
                    "status_code": 200,
                    "results": {"results": {"organic": LIVE_ORGANIC}},
                }
            }
        ]
    }
    fake_post, seen = _decodo_then_firecrawl(payload)
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://live.example/1" in out
    assert all("/v1/search" not in u for u in seen), f"fallback fired: {seen}"


def test_decodo_missing_results_key_falls_back(monkeypatch):
    """A body with no results list is a failure; its message is surfaced."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    fake_post, seen = _decodo_then_firecrawl({"message": "no task envelope"})
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://f.example" in out
    assert any("/v1/search" in u for u in seen)


def test_decodo_missing_credentials_falls_back(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.delenv("DECODO_API_TOKEN", raising=False)

    def fake_post(url, **kw):
        assert "/v1/search" in url
        return FakeResp(FIRECRAWL_OK)

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://f.example" in out


def test_decodo_token_is_sent_as_basic_auth(monkeypatch):
    """DECODO_API_TOKEN is used verbatim as the Basic credential."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=False))
    monkeypatch.setenv("DECODO_API_TOKEN", "abc123token")
    captured = {}

    def fake_post(url, **kw):
        captured["headers"] = kw["headers"]
        captured["auth"] = kw.get("auth")
        return FakeResp(_live_payload(LIVE_ORGANIC))

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)

    assert captured["headers"]["Authorization"] == "Basic abc123token"
    assert captured["auth"] is None, "Basic auth must come from the token header, not requests auth"
    assert "https://live.example/1" in out, out  # the request really succeeded


def test_username_password_env_vars_are_not_accepted(monkeypatch):
    """Username/password must not authenticate: DECODO_API_TOKEN is the only credential."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=False))
    monkeypatch.delenv("DECODO_API_TOKEN", raising=False)
    monkeypatch.setenv("DECODO_USERNAME", "u")
    monkeypatch.setenv("DECODO_PASSWORD", "p")
    called = []

    def fake_post(url, **kw):
        called.append(url)
        return FakeResp(_live_payload(LIVE_ORGANIC))

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)

    assert out.startswith("Search error:"), out
    assert "DECODO_API_TOKEN" in out
    assert not called, f"username/password must not reach the API: {called}"


def test_decodo_missing_token_reports_the_env_var(monkeypatch):
    monkeypatch.setenv("DECODO_API_TOKEN", "")  # present but empty
    results, err = search_mod._decodo_search("q", _settings(fallback=False).search, 5)

    assert results == []
    assert err and "DECODO_API_TOKEN" in err


def test_malformed_payload_returns_error_not_raise(monkeypatch):
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")

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


def test_limit_is_respected(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings())
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    organic = [
        {"title": "A", "url": "https://a.example", "desc": "a"},
        {"title": "B", "url": "https://b.example", "desc": "b"},
        {"title": "C", "url": "https://c.example", "desc": "c"},
    ]
    monkeypatch.setattr(
        search_mod.requests, "post", lambda url, **kw: FakeResp(_live_payload(organic))
    )
    out = search_mod._search("q", 2)
    assert out.count("https://") == 2
    assert "https://c.example" not in out


def test_top_level_errors_with_valid_results_keeps_results(monkeypatch):
    """A top-level error list beside usable results must not discard them."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    payload = {"errors": ["warn"], "results": [{"content": _live_content(LIVE_ORGANIC)}]}
    fake_post, seen = _decodo_then_firecrawl(payload)
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://live.example/1" in out
    assert all("/v1/search" not in u for u in seen), f"fallback fired: {seen}"


def test_top_level_message_with_empty_content_entry_falls_back(monkeypatch):
    """A provider error must not be masked by an empty-content task entry."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    payload = {"message": "rate limited", "results": [{"content": {}}]}
    fake_post, seen = _decodo_then_firecrawl(payload)
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://f.example" in out
    assert any("/v1/search" in u for u in seen)


@pytest.mark.parametrize(
    "payload",
    [
        {"results": [{"content": []}]},
        {"results": ["x"]},
    ],
)
def test_malformed_entries_are_failure_not_empty(monkeypatch, payload):
    """A non-empty results list with no usable entry must fall back."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    fake_post, seen = _decodo_then_firecrawl(payload)
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://f.example" in out, f"payload={payload!r} -> {out!r}"
    assert any("/v1/search" in u for u in seen), f"payload={payload!r} no fallback"


def test_content_empty_errors_key_is_empty_success(monkeypatch):
    """A present empty ``errors`` list is the documented success marker."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    fake_post, seen = _decodo_then_firecrawl({"results": [{"content": {"errors": []}}]})
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "No search results found" in out
    assert all("/v1/search" not in u for u in seen), f"fallback fired: {seen}"


def test_decodo_failure_with_default_fallback_off_does_not_call_firecrawl(monkeypatch):
    """The default (fallback off) surfaces the error instead of retrying."""
    s = ORASettings()
    s.search = SearchSettings(provider="decodo")  # fallback_to_firecrawl defaults False
    monkeypatch.setattr(search_mod, "load_config", lambda: s)
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    called: list[str] = []

    def fake_post(url, **kw):
        called.append(url)
        return FakeResp({"status": "failed", "status_code": 613, "message": "nope"})

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert out.startswith("Search error:"), out
    assert all("/v1/search" not in u for u in called), f"fallback fired: {called}"


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


def test_unknown_provider_warns_and_uses_firecrawl(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(provider="decodo "))
    monkeypatch.setenv("FIRECRAWL_API_URL", "http://selfhosted.example:3002")

    def fake_post(url, **kw):
        return FakeResp(FIRECRAWL_OK)

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    with pytest.warns(UserWarning, match="Unknown search provider"):
        out = search_mod._search("q", 5)
    assert "https://f.example" in out


def test_decodo_populated_shallower_organic_beats_empty_deep(monkeypatch):
    """An empty deepest organic list must not shadow a populated shallower one."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    content = {
        "results": {
            "parse_status_code": 12000,
            "results": {"organic": []},
            "organic": [{"title": "Shallow", "url": "https://shallow.example", "desc": "shallow"}],
        },
        "errors": [],
    }
    fake_post, seen = _decodo_then_firecrawl({"results": [{"content": content}]})
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://shallow.example" in out, out
    assert all("/v1/search" not in u for u in seen), f"fallback fired: {seen}"


def test_results_win_over_job_level_failure_signal(monkeypatch):
    """Pins results-win precedence: usable content beats a job-level failure."""
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(fallback=True))
    monkeypatch.setenv("DECODO_API_TOKEN", "test-token")
    payload = {
        "status": "failed",
        "status_code": 613,
        "message": "job failed but content present",
        "results": [{"content": _live_content(LIVE_ORGANIC, errors=[])}],
    }
    fake_post, seen = _decodo_then_firecrawl(payload)
    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    out = search_mod._search("q", 5)
    assert "https://live.example/1" in out, out
    assert all("/v1/search" not in u for u in seen), f"fallback fired: {seen}"


def test_firecrawl_sends_api_key_header(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(provider="firecrawl"))
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-secret")
    captured = {}

    def fake_post(url, **kw):
        captured["headers"] = kw["headers"]
        return FakeResp(FIRECRAWL_OK)

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    search_mod._search("q", 5)
    assert captured["headers"]["Authorization"] == "Bearer fc-secret"


def test_firecrawl_omits_auth_header_without_key(monkeypatch):
    monkeypatch.setattr(search_mod, "load_config", lambda: _settings(provider="firecrawl"))
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.delenv("ORA_SEARCH__FIRECRAWL_API_KEY", raising=False)
    captured = {}

    def fake_post(url, **kw):
        captured["headers"] = kw["headers"]
        return FakeResp(FIRECRAWL_OK)

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    search_mod._search("q", 5)
    assert "Authorization" not in captured["headers"]


def test_firecrawl_uses_settings_api_key(monkeypatch):
    """When the env var is unset, the config-file key is used."""
    s = ORASettings()
    s.search = SearchSettings(provider="firecrawl", firecrawl_api_key="from-settings")
    monkeypatch.setattr(search_mod, "load_config", lambda: s)
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    captured = {}

    def fake_post(url, **kw):
        captured["headers"] = kw["headers"]
        return FakeResp(FIRECRAWL_OK)

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    search_mod._search("q", 5)
    assert captured["headers"]["Authorization"] == "Bearer from-settings"


class TestSearchTool:
    def test_tool_has_name(self):
        assert web_search.name == "web_search"

    def test_tool_has_description(self):
        assert "search" in web_search.description.lower()
