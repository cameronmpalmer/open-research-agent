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


def _should_skip_url(url: str) -> bool:
    """Return True if the URL's domain is known to block automated scraping."""
    try:
        domain = urlparse(url).netloc.lower()
    except Exception:
        return False
    return domain in SKIP_DOMAINS


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
    }
    templates = angles.get(intensity, angles[2])
    return [t.format(query=query) for t in templates]


def researcher_node(
    state: ResearchState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Researcher: template-based queries, programmatic search/scrape.
    
    Synchronous node -- uses blocking HTTP calls inside LangGraph's sync execution.
    """
    settings = load_config()

    query = state.get("query", "")
    intensity = state.get("intensity", 2)
    query_count = {1: 1, 2: 3, 3: 5}.get(intensity, 3)

    queries = generate_search_queries(query, intensity)[:query_count]

    query_label = "query" if len(queries) == 1 else "queries"
    emit_progress(
        config,
        f"Researcher: generated {len(queries)} search {query_label}",
        kind="search",
    )

    from ora.tools.search import web_search
    from ora.tools.scrape import scrape_page
    from ora.tools.evaluate import evaluate_source

    sources, findings = [], []
    log: list[str] = []

    for q in queries:
        log.append(f"Search: {q}")
        emit_progress(config, f'Researcher: searching "{q}"', kind="search")
        r = web_search.invoke({"query": q})
        log.append(f"  Result: {len(r)} chars")
        search_failed = r.startswith("Search error") or r.startswith("Search failed")
        if search_failed:
            emit_progress(config, f'Researcher: search failed for "{q}"', kind="error")

        urls = re.findall(r'https?://[^\s<>"\')\]]+', r)
        log.append(f"  URLs found: {len(urls)}")
        url_label = "URL" if len(urls) == 1 else "URLs"
        emit_progress(
            config,
            f"Researcher: found {len(urls)} {url_label}",
            kind="info" if search_failed else "success",
        )

        scraped_count = 0
        for url in urls:  # walk all URLs until 2 successes, skipping hostile domains
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
            try:
                source = evaluate_source(url=url, content=c, source_type="unknown")
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
            findings.append(
                Finding(
                    claim=c[:500],
                    confidence=finding_confidence,  # type: ignore[arg-type]
                    supporting_sources=[url],
                )
            )
            scraped_count += 1
            if scraped_count >= 2:  # limit to 2 successful scrapes per query
                break

    if not findings:
        results_text = web_search.invoke({"query": query})
        findings.append(Finding(
            claim=f"No scraped content found. Raw search: {results_text[:300]}",
            confidence="Unknown",
        ))

    source_label = "source" if len(sources) == 1 else "sources"
    finding_label = "finding" if len(findings) == 1 else "findings"
    emit_progress(
        config,
        f"Researcher: finished with {len(sources)} {source_label} and {len(findings)} {finding_label}",
        kind="success",
    )

    return {
        "search_queries": queries,
        "sources": sources,
        "findings": findings,
        "messages": ["\n".join(log)],
    }
