"""Researcher agent node for LangGraph."""
import re
from typing import Any, Optional
from urllib.parse import urlparse
from langchain_core.runnables import RunnableConfig
from ora.state import ResearchState, Source, Finding
from ora.config import load_config, get_llm
from ora.progress import emit_progress
from ora.prompts import GAP_QUERY_PROMPT

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


def _prefilter_urls(
    urls: list[str],
    seen_urls: set[str],
) -> tuple[list[str], int, int]:
    """Filter URLs to remove already-seen and hostile-domain entries.

    Returns a tuple of (new_urls, dup_count, hostile_count).
    The caller receives only guaranteed-fresh, non-hostile URLs.
    """
    new_urls = []
    skipped_dup = 0
    skipped_hostile = 0
    for url in urls:
        normalized = _normalize_url_for_dedupe(url)
        if normalized in seen_urls:
            skipped_dup += 1
            continue
        if _should_skip_url(url):
            skipped_hostile += 1
            continue
        new_urls.append(url)
    return new_urls, skipped_dup, skipped_hostile


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


def _format_reviewer_feedback(state: ResearchState) -> str:
    """Extract reviewer feedback for the gap query prompt.

    Returns empty string if no reviewer verdict exists.
    """
    verdict = state.get("review_verdict")
    if verdict is None:
        return ""

    parts: list[str] = []
    if hasattr(verdict, 'blocking') and verdict.blocking:
        parts.append(
            "BLOCKING issues:\n" + "\n".join(f"- {b}" for b in verdict.blocking)
        )
    if hasattr(verdict, 'required') and verdict.required:
        parts.append(
            "REQUIRED improvements:\n" + "\n".join(f"- {r}" for r in verdict.required)
        )
    if hasattr(verdict, 'suggested') and verdict.suggested:
        parts.append(
            "SUGGESTED improvements:\n" + "\n".join(f"- {s}" for s in verdict.suggested)
        )
    if hasattr(verdict, 'contradicting_evidence_found') and verdict.contradicting_evidence_found:
        parts.append(
            "CONTRADICTING EVIDENCE found:\n"
            + "\n".join(f"- {c}" for c in verdict.contradicting_evidence_found)
        )
    return "\n\n".join(parts)


def generate_gap_queries_dynamic(
    query: str,
    intensity: int,
    sources: list[Source],
    reviewer_feedback: str,
    executed_queries: set[str],
    config: Optional[RunnableConfig] = None,
) -> list[str]:
    """Generate adaptive gap queries using the LLM.

    Uses context about what's been found and what the reviewer flagged to
    produce targeted, varied queries instead of repeating fixed templates.

    Falls back to template-based gap queries if the LLM call fails or if
    intensity is below 3 (where the cost isn't justified).
    """
    # For low intensities, template queries are sufficient.
    if intensity < 3:
        return generate_gap_queries(query, intensity)

    # Number of queries to request from the LLM.
    count = {3: 5, 4: 7, 5: 10}.get(intensity, 5)

    # Build source summary: titles + key topics from findings.
    source_lines: list[str] = []
    for s in sources[:30]:
        if s.title:
            source_lines.append(f"- {s.title} ({s.source_type})")
    source_summary = "\n".join(source_lines) if source_lines else "(no sources yet)"

    # Already executed queries (sample of up to 30 for the prompt; set is unordered,
    # so selection is arbitrary but diverse enough to prevent repeats).
    executed_sorted = sorted(executed_queries)
    recent = executed_sorted[-30:]
    already_run = "\n".join(f"- {q}" for q in recent) if recent else "(none yet)"

    try:
        settings = load_config()
        model_name = settings.models.researcher or settings.models.default
        llm = get_llm(model_name, temperature=0.8)
        prompt_text = GAP_QUERY_PROMPT.format(
            query=query,
            source_summary=source_summary,
            reviewer_feedback=reviewer_feedback or "(no reviewer feedback yet)",
            already_run=already_run,
            count=count,
        )
        response = llm.invoke(prompt_text)
        text = response.content if hasattr(response, 'content') else str(response)
    except Exception:
        # Fall back to template queries if LLM call fails.
        emit_progress(config, "Researcher: gap query LLM failed, using templates", kind="warning")
        return generate_gap_queries(query, intensity)

    # Parse: one query per line, strip numbering and bullets.
    queries = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip "1. ", "1) " prefixes only (not content hyphens).
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        # Strip bullet markers like "- " and "* " only when followed by non-digit,
        # so "-1 penalty" is preserved but "- something" is stripped.
        line = re.sub(r'^[-*]\s+(?!\d)', '', line)
        if line:
            queries.append(line)

    if not queries:
        emit_progress(config, "Researcher: gap query LLM returned no queries, using templates", kind="warning")
        return generate_gap_queries(query, intensity)

    return queries[:count]


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
    query: str = "",
    intensity: int = 2,
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

        # Normalize the URL for source URL tracking.
        # source.url remains as-is (the original), seen_urls uses the normalized variant.
        try:
            if intensity >= 3:
                # LLM-powered extraction + evaluation at high intensities.
                from ora.tools.extract import extract_and_evaluate
                source, extraction = extract_and_evaluate(
                    url=url,
                    title=url_titles.get(normalized_url, ""),
                    content=c,
                    source_type="unknown",
                    query=query,
                    config=config,
                )
                # Use the LLM-extracted summary as the claim (much richer than c[:500]).
                claim_text = extraction.summary if extraction.summary else c[:500]
                log.append(f"  Extracted: {len(extraction.key_claims)} claims, "
                           f"{len(extraction.recommendations)} recommendations, "
                           f"reliability={extraction.source_reliability}")
            else:
                # Heuristic evaluation for cost efficiency at low intensities.
                source = evaluate_source(
                    url=url,
                    title=url_titles.get(normalized_url, ""),
                    content=c,
                    source_type="unknown",
                )
                extraction = None
                claim_text = c[:500]
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
            extraction = None
            claim_text = c[:500] if c else ""
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
                claim=claim_text,
                confidence=finding_confidence,  # type: ignore[arg-type]
                supporting_sources=[url],
                extraction=extraction,
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
    Queries already executed across *all* researcher invocations are deduplicated
    so the reviewer sending execution back to researcher does not waste search calls.
    Gap queries after round 1 are generated dynamically by the LLM using context
    about found sources and reviewer feedback.
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

    # Dedup set: queries executed across ALL researcher invocations.
    # This prevents wasted search calls when the reviewer sends execution back.
    executed_q_set: set[str] = set(state.get("executed_queries", []))
    if executed_q_set:
        log.append(f"Resuming with {len(executed_q_set)} previously-executed queries deduped")

    # Reviewer feedback for targeted gap queries.
    reviewer_feedback = _format_reviewer_feedback(state)

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
            # Dynamic gap queries using LLM: adapt to what's been found and
            # what the reviewer flagged. Falls back to templates on failure.
            queries_for_round = generate_gap_queries_dynamic(
                query=query,
                intensity=intensity,
                sources=sources,
                reviewer_feedback=reviewer_feedback,
                executed_queries=executed_q_set,
                config=config,
            )

        # Filter out queries already executed in any prior invocation.
        fresh_queries = [q for q in queries_for_round if q not in executed_q_set]
        if not fresh_queries and round_num > 1:
            # All gap queries are duplicates -- try one more LLM generation
            # with explicit instruction to avoid repeats.
            emit_progress(
                config,
                "Researcher: all gap queries were duplicates, regenerating...",
                kind="warning",
            )
            queries_for_round = generate_gap_queries_dynamic(
                query=query,
                intensity=intensity,
                sources=sources,
                reviewer_feedback=reviewer_feedback,
                executed_queries=executed_q_set,
                config=config,
            )
            fresh_queries = [q for q in queries_for_round if q not in executed_q_set]

        if not fresh_queries:
            emit_progress(
                config,
                f"Researcher: no new queries to try after {round_num} rounds",
                kind="info",
            )
            break

        kind_label = "search" if round_num == 1 else "gap"
        query_label = "query" if len(fresh_queries) == 1 else "queries"
        emit_progress(
            config,
            f"Researcher: {len(fresh_queries)} {kind_label} {query_label} ({len(queries_for_round) - len(fresh_queries)} duplicates skipped)",
            kind="search",
        )

        for q in fresh_queries:
            if len(sources) >= min_sources:
                break

            executed_q_set.add(q)
            all_queries.append(q)
            log.append(f"Search: {q}")
            emit_progress(config, f'Researcher: searching "{q}"', kind="search")
            r = web_search.invoke({"query": q})
            log.append(f"  Result: {len(r)} chars")
            search_failed = r.startswith("Search error") or r.startswith("Search failed")
            if search_failed:
                emit_progress(config, f'Researcher: search failed for "{q}"', kind="error")

            raw_urls = re.findall(r'https?://[^\s<>"\')\]]+', r)
            url_titles = _extract_search_result_titles(r)
            raw_count = len(raw_urls)

            # Pre-filter: remove duplicates and hostile domains so the
            # scraper only processes guaranteed-fresh URLs.
            urls, skipped_dup, skipped_hostile = _prefilter_urls(raw_urls, seen_urls)
            log.append(
                f"  URLs found: {raw_count} total, {len(urls)} new"
                f" ({skipped_dup} dups, {skipped_hostile} hostile)"
            )

            url_label = "URL" if len(urls) == 1 else "URLs"
            dedup_note = ""
            if skipped_dup or skipped_hostile:
                dedup_note = f" (filtered: {skipped_dup} dups"
                if skipped_hostile:
                    dedup_note += f", {skipped_hostile} hostile"
                dedup_note += ")"
            emit_progress(
                config,
                f"Researcher: found {len(urls)} candidate {url_label}{dedup_note}",
                kind="info" if search_failed else "success",
            )

            if _scrape_and_collect(
                urls, params, max_content_chars, config, log,
                sources, findings, seen_urls, url_titles, min_sources=min_sources,
                query=query, intensity=intensity,
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
        "executed_queries": list(executed_q_set),
        "sources": sources[prior_source_count:],
        "findings": findings[prior_finding_count:],
        "research_status": research_status,
        "messages": ["\n".join(log)],
    }
