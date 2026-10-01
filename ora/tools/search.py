"""Web search tool for LangChain, with a Decodo primary and Firecrawl fallback."""

import json
import os
from base64 import b64encode

import requests
from langchain_core.tools import tool

from ora.config import load_config

_MAX_RESULTS = 5


def _decodo_search(query: str, s) -> tuple[list[dict], str | None]:
    """Return (results, error). results is [] on any failure."""
    username = os.environ.get(s.decodo_username_env, "")
    password = os.environ.get(s.decodo_password_env, "")
    if not username or not password:
        return [], f"decodo credentials missing ({s.decodo_username_env}/{s.decodo_password_env})"
    body = json.dumps(
        {
            "target": "google_search",
            "query": query,
            "domain": s.decodo_domain,
            "locale": s.decodo_locale,
            "parse": True,
            "device_type": "desktop_chrome",
        }
    ).encode()
    try:
        resp = requests.post(
            s.decodo_api_url,
            data=body,
            headers={
                "Authorization": f"Basic {b64encode(f'{username}:{password}'.encode()).decode()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=30,
        )
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        return [], f"decodo request failed: {e!s}"

    if data.get("status") == "failed":
        return [], f"decodo status {data.get('status_code')}: {data.get('message')}"

    out: list[dict] = []
    for entry in data.get("results", []):
        content = entry.get("content", {}) or {}
        outer = content.get("results", {}) or {}
        organic = outer.get("organic") or content.get("organic") or []
        for item in organic[:_MAX_RESULTS]:
            out.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                }
            )
    return out[:_MAX_RESULTS], None


def _firecrawl_search(query: str, s) -> tuple[list[dict], str | None]:
    try:
        resp = requests.post(
            f"{s.firecrawl_api_url}/v1/search",
            json={"query": query, "limit": _MAX_RESULTS},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        return [], f"firecrawl request failed: {e!s}"
    if not data.get("success"):
        return [], f"firecrawl search failed: {data}"
    out = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "description": item.get("description", ""),
        }
        for item in (data.get("data") or [])[:_MAX_RESULTS]
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
        results, err = _decodo_search(query, s.search)
        if results:
            return _format(results)
        if err is None:
            return "No search results found."
        if not s.search.fallback_to_firecrawl:
            return f"Search error: {err}"
        fb, fb_err = _firecrawl_search(query, s.search)
        if fb:
            return _format(fb)
        if fb_err is None:
            return "No search results found."
        return f"Search error: decodo={err}; firecrawl={fb_err}"
    results, err = _firecrawl_search(query, s.search)
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
