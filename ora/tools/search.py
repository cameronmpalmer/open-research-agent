"""Web search tool for LangChain, with a Decodo primary and Firecrawl fallback."""

import json
import os

import requests
from langchain_core.tools import tool

from ora.config import SearchSettings, load_config

_MAX_RESULTS = 5


def _decodo_search(
    query: str, settings: SearchSettings, limit: int = _MAX_RESULTS
) -> tuple[list[dict], str | None]:
    """Return (results, error). results is [] on any failure."""
    username = os.environ.get(settings.decodo_username_env, "")
    password = os.environ.get(settings.decodo_password_env, "")
    if not username or not password:
        return [], (
            f"decodo credentials missing "
            f"({settings.decodo_username_env}/{settings.decodo_password_env})"
        )
    body = json.dumps(
        {
            "target": "google_search",
            "query": query,
            "domain": settings.decodo_domain,
            "locale": settings.decodo_locale,
            "parse": True,
            "device_type": "desktop_chrome",
        }
    ).encode()
    try:
        resp = requests.post(
            settings.decodo_api_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            auth=(username, password),
            timeout=30,
        )
        if resp.status_code >= 400:
            return [], f"decodo HTTP {resp.status_code}"
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        return [], f"decodo request failed: {e!s}"

    if not isinstance(data, dict):
        return [], f"decodo malformed response: {type(data).__name__}"
    if data.get("status") == "failed":
        return [], f"decodo status {data.get('status_code')}: {data.get('message')}"
    if "results" not in data:
        return [], "decodo malformed response: no results key"

    entries = data["results"]
    if not isinstance(entries, list):
        return [], f"decodo malformed results: {type(entries).__name__}"

    out: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        content = entry.get("content") or {}
        if not isinstance(content, dict):
            continue
        # Decodo's parse:true response nests organic results by depth, and the
        # live API (verified 2026-10-01) uses the two-level shape
        # content.results.results.organic. Older/other responses use a single
        # level (content.results.organic) or put organic directly on content,
        # so accept all three, preferring the deeper (live) shape.
        outer = content.get("results") or {}
        if not isinstance(outer, dict):
            outer = {}
        inner = outer.get("results")
        organic: object = None
        for candidate in (
            inner.get("organic") if isinstance(inner, dict) else None,
            outer.get("organic"),
            content.get("organic"),
        ):
            if isinstance(candidate, list) and candidate:
                organic = candidate  # first non-empty list wins
                break
            if organic is None and isinstance(candidate, list):
                organic = candidate  # remember an empty list as zero results
        if not isinstance(organic, list):
            organic = []
        for item in organic[:limit]:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    # Live Decodo organic items use "desc"; some fixtures and
                    # fallback shapes use "description".
                    "description": item.get("description") or item.get("desc") or "",
                }
            )
    return out[:limit], None


def _firecrawl_search(
    query: str, settings: SearchSettings, limit: int = _MAX_RESULTS
) -> tuple[list[dict], str | None]:
    url = os.environ.get("FIRECRAWL_API_URL", settings.firecrawl_api_url)
    try:
        resp = requests.post(
            f"{url}/v1/search",
            json={"query": query, "limit": limit},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if resp.status_code >= 400:
            return [], f"firecrawl HTTP {resp.status_code}"
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        return [], f"firecrawl request failed: {e!s}"
    if not isinstance(data, dict):
        return [], f"firecrawl malformed response: {type(data).__name__}"
    if not data.get("success"):
        return [], f"firecrawl search failed: {data}"
    if "data" not in data:
        return [], "firecrawl malformed response: no data key"
    items = data["data"]
    if not isinstance(items, list):
        return [], f"firecrawl malformed data: {type(items).__name__}"
    out = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "description": item.get("description", ""),
        }
        for item in items[:limit]
        if isinstance(item, dict)
    ]
    return out, None


def _format(results: list[dict]) -> str:
    if not results:
        return "No search results found."
    formatted = []
    for i, item in enumerate(results, 1):
        url = item.get("url", "")
        title = item.get("title", "Untitled")
        snippet = (item.get("description", "") or "")[:300]
        formatted.append(f"{i}. [{title}]({url})\n   {snippet}")
    return "\n\n".join(formatted)


def _search(query: str, limit: int = _MAX_RESULTS) -> str:
    """Provider dispatch. Returns formatted results or an explicit error string.

    Distinguishes "provider failed" (err is not None) from "provider succeeded
    with zero results" (err is None, results empty). Only the former triggers a
    fallback; the latter is a legitimate empty answer.
    """
    s = load_config()
    provider = (s.search.provider or "firecrawl").lower()
    if provider == "decodo":
        results, err = _decodo_search(query, s.search, limit)
        if results:
            return _format(results)
        if err is None:
            return "No search results found."
        if not s.search.fallback_to_firecrawl:
            return f"Search error: {err}"
        fb, fb_err = _firecrawl_search(query, s.search, limit)
        if fb:
            return _format(fb)
        if fb_err is None:
            return "No search results found."
        return f"Search error: decodo={err}; firecrawl={fb_err}"
    results, err = _firecrawl_search(query, s.search, limit)
    if results:
        return _format(results)
    if err is None:
        return "No search results found."
    return f"Search error: {err}"


@tool
def web_search(query: str) -> str:
    """Search the web.

    Args:
        query: The search query string.

    Returns:
        Search results as formatted text with URLs and snippets.
    """
    try:
        return _search(query)
    except Exception as e:  # noqa: BLE001
        return f"Search error: {e!s}"
