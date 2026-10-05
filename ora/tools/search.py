"""Web search tool for LangChain, with a Decodo primary and Firecrawl fallback.

The Decodo classification follows the documented Web Scraping API contract
(https://help.decodo.com/docs/web-scraping-api-response-codes and
https://help.decodo.com/docs/web-scraping-api-status-codes, verified live
2026-10-02):

* HTTP: ``200`` success; ``204`` job not complete, retry in a few seconds;
  ``>= 400`` failure (``500``/``524`` are not billed, but any failure still
  falls back when the fallback is enabled).
* Body: top-level ``status == "failed"`` is a failure (for example the HTTP 200
  ``status_code: 613`` envelope a no-results query returns; 613 is a body
  status code, not an HTTP status). A top-level ``status_code`` outside
  ``{200, 202}`` is also a failure.
* Per task: ``parse_status_code`` in ``{12000, 12004, 12005}`` is a usable
  parse. When it is absent, ``content.status_code`` in ``{200, 12000}`` is the
  fallback signal. A non-empty ``content.errors`` is a failure.
* A successful ``parse: true`` ``google_search`` nests organic results at
  ``content.results.results.organic`` (older shapes nest one level shallower or
  put ``organic`` on ``content``). Organic items use ``desc`` for the snippet.

Classification precedence, applied once after parsing: parsed organic results
win (a job-level failure signal beside usable content is ignored, since a
needless failure is worse than trusting real content); otherwise a collected
failure signal fails and, when enabled, triggers the Firecrawl fallback;
otherwise the result is a legitimate empty parse and never falls back.
"""

import json
import os
import time

import requests
from langchain_core.tools import tool

from ora.config import SearchSettings, load_config

_MAX_RESULTS = 5

# Decodo can answer HTTP 204 ("job not complete; retry in a few seconds") while
# an async job finishes. Retry a small bounded number of times.
_MAX_HTTP_CALLS = 3
_RETRY_SLEEP_SECONDS = 3

# Documented parser status codes meaning the parse is usable (complete or
# partial). Everything else present (12002, 12003, 12006-12009) is a failure.
# https://help.decodo.com/docs/web-scraping-api-status-codes
_USABLE_PARSE_STATUS = frozenset({12000, 12004, 12005})
# Documented top-level body status codes meaning the job itself succeeded.
_TOP_LEVEL_SUCCESS_STATUS = frozenset({200, 202})
# Fallback success set for content.status_code when parse_status_code is absent.
_USABLE_CONTENT_STATUS = frozenset({200, 12000})


def _non_empty_error(value: object) -> bool:
    """True when an error field actually carries an error.

    Decodo error fields are absent, an empty list/string (success), or a
    populated list/string (failure). Non-empty dicts are treated as failures
    too, defensively.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return False


def _parse_status_is_usable(value: object) -> bool | None:
    """Usable per ``parse_status_code``: None when absent, else True/False."""
    if value is None:
        return None
    try:
        return int(value) in _USABLE_PARSE_STATUS  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _find_parse_status(content: dict) -> object | None:
    """The parser status, looked up at content level or in its results dicts.

    Observed live (2026-10-02): a successful ``google_search`` carries it at
    ``content.results.parse_status_code``. Some shapes put it directly on
    ``content`` (or one level deeper), so check all three.
    """
    candidates = [content.get("parse_status_code")]
    outer = content.get("results")
    if isinstance(outer, dict):
        candidates.append(outer.get("parse_status_code"))
        inner = outer.get("results")
        if isinstance(inner, dict):
            candidates.append(inner.get("parse_status_code"))
    for value in candidates:
        if value is not None:
            return value
    return None


def _content_status_is_usable(value: object) -> bool:
    """Fallback usability from content.status_code when parse_status_code is absent.

    A missing status is treated as usable (not reported is not a failure).
    """
    if value is None:
        return True
    try:
        return int(value) in _USABLE_CONTENT_STATUS  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _top_level_status_is_ok(value: object) -> bool:
    """Whether a present top-level status_code is a documented success."""
    if value is None:
        return True
    try:
        return int(value) in _TOP_LEVEL_SUCCESS_STATUS  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _extract_organic(content: dict) -> list:
    """The first non-empty organic list among the nested containers, else [].

    Prefer the deepest populated list: an empty ``content.results.results.organic``
    must not shadow a populated ``content.results.organic`` or ``content.organic``.
    """
    outer = content.get("results")
    inner = outer.get("results") if isinstance(outer, dict) else None
    for candidate in (
        inner.get("organic") if isinstance(inner, dict) else None,
        outer.get("organic") if isinstance(outer, dict) else None,
        content.get("organic"),
    ):
        if isinstance(candidate, list) and candidate:
            return candidate
    return []


def _decodo_search(
    query: str, settings: SearchSettings, limit: int = _MAX_RESULTS
) -> tuple[list[dict], str | None]:
    """Return (results, error). results is [] on any failure.

    Precedence is results-win: parsed organic results are returned even when a
    top-level job-level failure signal (``status == "failed"`` or a non-success
    ``status_code``) is also present. A failure is returned only when nothing
    usable parsed; otherwise the parse is a legitimate empty result
    (``([], None)``) and never triggers a fallback.
    """
    token = os.environ.get(settings.decodo_token_env, "").strip()
    if not token:
        return [], f"decodo credentials missing ({settings.decodo_token_env})"
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
        resp = None
        for attempt in range(_MAX_HTTP_CALLS):
            resp = requests.post(
                settings.decodo_api_url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    # Decodo issues a single Basic-auth credential (the API
                    # Playground token), which is the only accepted credential.
                    "Authorization": f"Basic {token}",
                },
                timeout=30,
            )
            if resp.status_code != 204:
                break
            if attempt < _MAX_HTTP_CALLS - 1:
                # Documented: 204 means the job is not complete yet; retry.
                time.sleep(_RETRY_SLEEP_SECONDS)
        if resp is not None and resp.status_code == 204:
            return [], f"decodo HTTP 204 (job not complete after {_MAX_HTTP_CALLS} calls)"
        if resp is None:
            return [], "decodo request failed: no response"
        if resp.status_code >= 400:
            err = f"decodo HTTP {resp.status_code}"
            try:
                err_body = resp.json()
            except Exception:  # noqa: BLE001
                err_body = None
            if isinstance(err_body, dict) and _non_empty_error(err_body.get("message")):
                err += f": {err_body.get('message')}"
            return [], err
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        return [], f"decodo request failed: {e!s}"

    if not isinstance(data, dict):
        return [], f"decodo malformed response: {type(data).__name__}"
    if "results" not in data:
        # No task envelope: the documented no-results (613) and other error
        # bodies put status/message here, so surface them rather than reading
        # the body as an empty answer.
        if data.get("status") == "failed":
            return [], f"decodo status {data.get('status_code')}: {data.get('message')}"
        if _non_empty_error(data.get("message")):
            return [], f"decodo message: {data.get('message')}"
        return [], "decodo malformed response: no results key"

    entries = data["results"]
    if not isinstance(entries, list):
        return [], f"decodo malformed results: {type(entries).__name__}"

    # Collect organic results and failure signals while parsing, then apply the
    # precedence once, after the loop. One decision point keeps the two
    # directions (results never fall back, failures always do) from drifting.
    out: list[dict] = []
    failures: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            failures.append("decodo task entry not an object")
            continue
        content = entry.get("content")
        if not isinstance(content, dict):
            # Observed live: a missing Authorization header returns content as a
            # raw HTML string, which is not a parse result.
            failures.append("decodo content not an object")
            continue
        # Recorded, not short-circuited: valid organic results take precedence.
        if _non_empty_error(content.get("errors")):
            failures.append(f"decodo content error: {content.get('errors')}")
        parse_status = _find_parse_status(content)
        parse_usable = _parse_status_is_usable(parse_status)
        if parse_usable is False:
            failures.append(f"decodo parse_status_code {parse_status}")
        elif parse_usable is None and not _content_status_is_usable(content.get("status_code")):
            failures.append(f"decodo content status {content.get('status_code')}")
        outer = content.get("results")
        if isinstance(outer, dict) and _non_empty_error(outer.get("errors")):
            failures.append(f"decodo results error: {outer.get('errors')}")
        for item in _extract_organic(content)[:limit]:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    # Live Decodo organic items use "desc"; older fixtures used
                    # "description", kept as a fallback key.
                    "description": item.get("desc") or item.get("description") or "",
                }
            )

    # The single classification decision, in precedence order.
    # 1. Parsed organic results win; warnings/error markers beside them are
    #    ignored, so a usable search is never thrown away or double-billed.
    if out:
        return out[:limit], None
    # 2. A provider failure signal: an explicit error so _search can fall back.
    if data.get("status") == "failed":
        return [], f"decodo status {data.get('status_code')}: {data.get('message')}"
    if not _top_level_status_is_ok(data.get("status_code")):
        return [], f"decodo status {data.get('status_code')}: {data.get('message')}"
    if failures:
        return [], failures[0]
    # Defensive extra signals, undocumented but harmless: only fire when nothing
    # usable was parsed, so they cannot discard a good result.
    if _non_empty_error(data.get("errors")):
        return [], f"decodo error: {data.get('errors')}"
    if _non_empty_error(data.get("message")):
        return [], f"decodo message: {data.get('message')}"
    # 3. No results and no failure signal: a legitimately empty parse.
    return [], None


def _firecrawl_search(
    query: str, settings: SearchSettings, limit: int = _MAX_RESULTS
) -> tuple[list[dict], str | None]:
    url = os.environ.get("FIRECRAWL_API_URL", settings.firecrawl_api_url)
    # Strip trailing slashes so a URL like "http://host:3002/" does not build
    # a "//v1/search" path.
    url = url.rstrip("/")
    # Firecrawl cloud requires the key; self-hosted installs run without one, so
    # only send the header when a key is configured.
    api_key = os.environ.get("FIRECRAWL_API_KEY") or settings.firecrawl_api_key
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.post(
            f"{url}/v1/search",
            json={"query": query, "limit": limit},
            headers=headers,
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
    elif provider != "firecrawl":
        # An unrecognized provider (e.g. a typo like "decodo ") must not
        # silently disable itself; name it and fall back, mirroring the LLM
        # provider registry's handling of unknown prefixes.
        import warnings

        warnings.warn(
            f"Unknown search provider '{s.search.provider}'; falling back to 'firecrawl'.",
            stacklevel=2,
        )
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
