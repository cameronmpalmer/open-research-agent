"""Researcher agent node for LangGraph."""
import re
from typing import Any, Optional
from urllib.parse import urlparse
from langchain_core.runnables import RunnableConfig
from ora.state import ResearchState, Source, Finding
from ora.config import load_config
from ora.progress import emit_progress

# Domains known to block or heavily rate-limit automated scraping.
# The researcher will skip these and try other results instead.
SKIP_DOMAINS = frozenset({
    "reddit.com",
    "www.reddit.com",
    "medium.com",
    "x.com",
    "twitter.com",
    "linkedin.com",
    "www.linkedin.com",
    "instagram.com",
    "facebook.com",
    "www.facebook.com",
    "tiktok.com",
    "youtube.com",
    "www.youtube.com",
    "quora.com",
    "www.quora.com",
})


LEVEL_PARAMS = {
    # max_rounds is a safety cap; the loop stops earlier when min_sources is reached.
    1: {"min_sources": 3,  "max_rounds": 5,  "urls_per_query": 5, "scrapes_per_query": 3, "max_content_chars": 8000},
    2: {"min_sources": 8,  "max_rounds": 5,  "urls_per_query": 5, "scrapes_per_query": 3, "max_content_chars": 8000},
    3: {"min_sources": 15, "max_rounds": 7,  "urls_per_query": 5, "scrapes_per_query": 3, "max_content_chars": 10000},
    4: {"min_sources": 50, "max_rounds": 10, "urls_per_query": 8, "scrapes_per_query": 4, "max_content_chars": 12000},
    5: {"min_sources": 100,"max_rounds": 10, "urls_per_query": 10,"scrapes_per_query": 5, "max_content_chars": 16000},
}


def _should_skip_url(url: str) -> bool:
    """Return True if the URL's domain is known to block automated scraping."""
    try:
        domain = urlparse(url).netloc.lower()
    except Exception:
        return False
    return domain in SKIP_DOMAINS


def _normalize_url_for_dedupe(url: str) -> str:
    """Normalize URL enough to avoid duplicate source entries."""
    try:
        parsed = urlparse(url)
    except Exception:
        return url.rstrip("/")

    path = parsed.path.rstrip("/")
    scheme = "https" if parsed.scheme.lower() in {"http", "https"} else parsed.scheme.lower()
    normalized = parsed._replace(
        scheme=scheme,
        netloc=parsed.netloc.lower(),
        path=path,
        fragment="",
    )
    return normalized.geturl()


def _extract_search_result_titles(search_results: str) -> dict[str, str]:
    """Extract URL -> title mappings from markdown-formatted search results."""
    titles: dict[str, str] = {}
    for title, url in re.findall(r'\[([^\]]+)\]\((https?://[^\s)]+)\)', search_results):
        titles[_normalize_url_for_dedupe(url)] = title.strip()
    return titles


def generate_search_queries(query: str, intensity: int) -> list[str]:
    """Generate search queries based on intensity level."""
    angles = {
        1: ["{query}"],
        2: ["{query}", "{query} latest", 'opposing view on {query}'],
        3: [
            "{query}",
            "{query} latest research 2025 2026",
            'critique of {query}',
            '"{query}" expert analysis',
            '{query} vs alternatives',
            '{query} best practices',
            'problems with {query}',
        ],
        4: [
            "{query}",
            "{query} latest research 2025 2026",
            'critique of {query}',
            '"{query}" expert analysis',
            '{query} vs alternatives',
            '{query} best practices',
            'problems with {query}',
            "{query} detailed analysis",
            "recent developments in {query}",
            "opposing view on {query}",
            "industry perspective on {query}",
            "academic perspective on {query}",
        ],
        5: [
            "{query}",
            "{query} latest research 2025 2026",
            'critique of {query}',
            '"{query}" expert analysis',
            '{query} vs alternatives',
            '{query} best practices',
            'problems with {query}',
            "{query} detailed analysis",
            "recent developments in {query}",
            "opposing view on {query}",
            "industry perspective on {query}",
            "academic perspective on {query}",
            "{query} statistics and data",
            "{query} case studies",
            "limitations of {query}",
            "controversies about {query}",
        ],
    }
    templates = angles.get(intensity, angles[2])
    return [t.format(query=query) for t in templates]


def generate_gap_queries(query: str, intensity: int) -> list[str]:
    """Generate gap-targeted queries when source counts fall short."""
    bases = {
        2: [
            "{query} detailed analysis",
            "key aspects of {query}",
        ],
        3: [
            "{query} detailed analysis",
            "key aspects of {query}",
            "expert review of {query}",
        ],
        4: [
            "{query} detailed analysis",
            "key aspects of {query}",
            "expert review of {query}",
            "recent developments in {query}",
            "opposing view on {query}",
        ],
        5: [
            "{query} detailed analysis",
            "key aspects of {query}",
            "expert review of {query}",
            "recent developments in {query}",
            "opposing view on {query}",
            "{query} statistics and data",
            "{query} case studies",
        ],
    }
    templates = bases.get(intensity, bases[2])
    return [t.format(query=query) for t in templates]


def _scrape_and_collect(
    urls: list[str],
    params: dict,
    max_content_chars: int,
    config: Optional[RunnableConfig],
    log: list[str],
    sources: list,
    findings: list,
    seen_urls: set[str],
    url_titles: dict[str, str],
    *_,
    min_sources: int,
) -> bool:
    """Scrape URLs up to the per-query cap, appending to sources/findings.

    Returns True if we've hit the overall min_sources target and the caller
    should stop further work.
    """
    from ora.tools.scrape import scrape_page
    from ora.tools.evaluate import evaluate_source

    scraped_this_query = 0
    for url in urls[:params["urls_per_query"]]:
        if len(sources) >= min_sources:
            return True

        normalized_url = _normalize_url_for_dedupe(url)
        if normalized_url in seen_urls:
            display_url = url.replace("https://", "").replace("http://", "")[:80]
            log.append(f"  Skipping duplicate source URL: {url[:80]}")
            emit_progress(config, f"Researcher: skipping duplicate {display_url}", kind="info")
            continue

        if _should_skip_url(url):
            display_url = url.replace("https://", "").replace("http://", "")[:80]
            log.append(f"  Skipping known-hostile domain: {url[:80]}")
            emit_progress(config, f"Researcher: skipping {display_url} (hostile domain)", kind="info")
            continue

        display_url = url.replace("https://", "").replace("http://", "")[:80]
        emit_progress(config, f"Researcher: scraping {display_url}", kind="scrape")
        c = scrape_page.invoke({"url": url})
        is_error = (
            c.startswith("Scrape error") or
            c.startswith("Scrape failed") or
            c.startswith("No content extracted")
        )
        log.append(f"  Scraped: {len(c)} chars from {url[:60]} {'(FAIL)' if is_error else ''}")
        if is_error:
            emit_progress(config, f"Researcher: scrape failed for {display_url}", kind="error")
            continue

        emit_progress(config, f"Researcher: scraped {len(c)} chars from {display_url}", kind="success")

        c = c[:max_content_chars]

        try:
            source = evaluate_source(
                url=url,
                title=url_titles.get(normalized_url, ""),
                content=c,
                source_type="unknown",
            )
        except Exception as e:
            log.append(f"  Source eval failed from {url[:60]}: {e}")
            emit_progress(config, f"Researcher: source evaluation failed for {display_url}", kind="error")
            source = Source(
                url=url,
                title="",
                source_type="unknown",
                overall_reliability="Low",
                notes=f"Source evaluation failed: {e}",
            )
        else:
            emit_progress(config, "Researcher: evaluated source reliability", kind="info")

        finding_confidence = {
            "High": "High",
            "Medium": "Moderate",
            "Low": "Low",
        }.get(source.overall_reliability, "Unknown")
        sources.append(source)
        seen_urls.add(normalized_url)
        findings.append(
            Finding(
                claim=c[:500],
                confidence=finding_confidence,  # type: ignore[arg-type]
                supporting_sources=[url],
            )
        )
        scraped_this_query += 1
        if scraped_this_query >= params["scrapes_per_query"]:
            break

    return len(sources) >= min_sources


def researcher_node(
    state: ResearchState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Researcher: iterate until min_sources target is reached.

    Synchronous node -- uses blocking HTTP calls inside LangGraph's sync execution.
    Per-query caps prevent one search angle from flooding results, but the overall
    loop continues generating queries and scraping until min_sources is hit (or the
    safety max_rounds cap is exceeded).
    """
    settings = load_config()

    query = state.get("query", "")
    intensity = state.get("intensity", 2)
    params = LEVEL_PARAMS.get(intensity, LEVEL_PARAMS[2])
    min_sources = params["min_sources"]
    max_rounds = params["max_rounds"]
    max_content_chars = params["max_content_chars"]

    from ora.tools.search import web_search

    sources: list = list(state.get("sources")) if state.get("sources") else []
    findings: list = list(state.get("findings")) if state.get("findings") else []
    prior_source_count = len(sources)
    prior_finding_count = len(findings)
    seen_urls = {
        _normalize_url_for_dedupe(source.url)
        for source in sources
        if hasattr(source, "url")
    }
    log: list[str] = []
    all_queries: list[str] = []
    round_num = 0

    while len(sources) < min_sources and round_num < max_rounds:
        round_num += 1
        emit_progress(
            config,
            f"Researcher: round {round_num} (have {len(sources)}, need {min_sources})",
            kind="info",
        )

        if round_num == 1:
            queries_for_round = list(generate_search_queries(query, intensity))
        else:
            queries_for_round = list(generate_gap_queries(query, intensity))

        if not queries_for_round:
            emit_progress(config, "Researcher: no more queries to try", kind="info")
            break

        kind_label = "search" if round_num == 1 else "gap"
        query_label = "query" if len(queries_for_round) == 1 else "queries"
        emit_progress(
            config,
            f"Researcher: generated {len(queries_for_round)} {kind_label} {query_label}",
            kind="search",
        )

        for q in queries_for_round:
            if len(sources) >= min_sources:
                break

            all_queries.append(q)
            log.append(f"Search: {q}")
            emit_progress(config, f'Researcher: searching "{q}"', kind="search")
            r = web_search.invoke({"query": q})
            log.append(f"  Result: {len(r)} chars")
            search_failed = r.startswith("Search error") or r.startswith("Search failed")
            if search_failed:
                emit_progress(config, f'Researcher: search failed for "{q}"', kind="error")

            urls = re.findall(r'https?://[^\s<>"\')\]]+', r)
            url_titles = _extract_search_result_titles(r)
            log.append(f"  URLs found: {len(urls)}")
            url_label = "URL" if len(urls) == 1 else "URLs"
            emit_progress(
                config,
                f"Researcher: found {len(urls)} {url_label}",
                kind="info" if search_failed else "success",
            )

            if _scrape_and_collect(
                urls, params, max_content_chars, config, log,
                sources, findings, seen_urls, url_titles, min_sources=min_sources,
            ):
                break

    if not findings:
        results_text = web_search.invoke({"query": query})
        findings.append(Finding(
            claim=f"No scraped content found. Raw search: {results_text[:300]}",
            confidence="Unknown",
        ))

    research_status = "final" if len(sources) >= min_sources else "interim"

    source_label = "source" if len(sources) == 1 else "sources"
    finding_label = "finding" if len(findings) == 1 else "findings"
    emit_progress(
        config,
        f"Researcher: finished with {len(sources)} {source_label} and {len(findings)} {finding_label} ({research_status})",
        kind="success",
    )

    return {
        "search_queries": all_queries,
        "sources": sources[prior_source_count:],
        "findings": findings[prior_finding_count:],
        "research_status": research_status,
        "messages": ["\n".join(log)],
    }
