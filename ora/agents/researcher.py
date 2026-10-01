"""Researcher agent node for LangGraph."""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlparse

from langchain_core.runnables import RunnableConfig

from ora.config import get_llm, get_researcher_model, load_config
from ora.progress import emit_progress
from ora.prompts import GAP_QUERY_PROMPT, ITEM_GAP_QUERY_PROMPT
from ora.state import Finding, ResearchState, Source

# Domains known to block or heavily rate-limit automated scraping.
# The researcher will skip these and try other results instead.
SKIP_DOMAINS = frozenset(
    {
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
    }
)


LEVEL_PARAMS = {
    # max_rounds is a safety cap; the loop stops earlier when min_sources is reached.
    1: {
        "min_sources": 3,
        "max_rounds": 5,
        "urls_per_query": 5,
        "scrapes_per_query": 3,
        "max_content_chars": 8000,
    },
    2: {
        "min_sources": 8,
        "max_rounds": 5,
        "urls_per_query": 5,
        "scrapes_per_query": 3,
        "max_content_chars": 8000,
    },
    3: {
        "min_sources": 15,
        "max_rounds": 7,
        "urls_per_query": 5,
        "scrapes_per_query": 3,
        "max_content_chars": 10000,
    },
    4: {
        "min_sources": 50,
        "max_rounds": 10,
        "urls_per_query": 8,
        "scrapes_per_query": 4,
        "max_content_chars": 12000,
    },
    5: {
        "min_sources": 100,
        "max_rounds": 10,
        "urls_per_query": 10,
        "scrapes_per_query": 5,
        "max_content_chars": 16000,
    },
}


def _should_skip_url(url: str) -> bool:
    """Return True if the URL's domain is known to block automated scraping."""
    try:
        domain = urlparse(url).netloc.lower()
    except Exception:  # noqa: BLE001
        return False
    return domain in SKIP_DOMAINS


def _normalize_url_for_dedupe(url: str) -> str:
    """Normalize URL enough to avoid duplicate source entries."""
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001
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
    for title, url in re.findall(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", search_results):
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
        2: ["{query}", "{query} latest", "opposing view on {query}"],
        3: [
            "{query}",
            "{query} latest research 2025 2026",
            "critique of {query}",
            '"{query}" expert analysis',
            "{query} vs alternatives",
            "{query} best practices",
            "problems with {query}",
        ],
        4: [
            "{query}",
            "{query} latest research 2025 2026",
            "critique of {query}",
            '"{query}" expert analysis',
            "{query} vs alternatives",
            "{query} best practices",
            "problems with {query}",
            "{query} detailed analysis",
            "recent developments in {query}",
            "opposing view on {query}",
            "industry perspective on {query}",
            "academic perspective on {query}",
        ],
        5: [
            "{query}",
            "{query} latest research 2025 2026",
            "critique of {query}",
            '"{query}" expert analysis',
            "{query} vs alternatives",
            "{query} best practices",
            "problems with {query}",
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
    if hasattr(verdict, "blocking") and verdict.blocking:
        parts.append("BLOCKING issues:\n" + "\n".join(f"- {b}" for b in verdict.blocking))
    if hasattr(verdict, "required") and verdict.required:
        parts.append("REQUIRED improvements:\n" + "\n".join(f"- {r}" for r in verdict.required))
    if hasattr(verdict, "suggested") and verdict.suggested:
        parts.append("SUGGESTED improvements:\n" + "\n".join(f"- {s}" for s in verdict.suggested))
    if hasattr(verdict, "contradicting_evidence_found") and verdict.contradicting_evidence_found:
        parts.append(
            "CONTRADICTING EVIDENCE found:\n"
            + "\n".join(f"- {c}" for c in verdict.contradicting_evidence_found)
        )
    return "\n\n".join(parts)


def _parse_query_lines(text: str, limit: int | None = None) -> list[str]:
    """Parse one-query-per-line LLM output into a query list.

    Uses regex markers so content that starts with digits or hyphens
    survives: strips "1. ", "1) ", "- ", and "* " list markers only, not
    content-leading digits (e.g. a year) or content hyphens (e.g. "-1
    penalty"). When limit is given, returns at most that many queries
    (matching the slicing generate_gap_queries_dynamic applied after
    parsing).
    """
    queries = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip "1. " and "1) " numbering prefixes only (not bare digits).
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        # Strip bullet markers "- " and "* " only when not followed by a
        # digit, so "-1 penalty" is preserved but "- something" is stripped.
        line = re.sub(r"^[-*]\s+(?!\d)", "", line)
        if line:
            queries.append(line)
    if limit is not None:
        return queries[:limit]
    return queries


def generate_gap_queries_dynamic(
    query: str,
    intensity: int,
    sources: list[Source],
    reviewer_feedback: str,
    executed_queries: set[str],
    config: RunnableConfig | None = None,
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
        model_name = get_researcher_model(settings)
        llm = get_llm(model_name, temperature=0.8)
        prompt_text = GAP_QUERY_PROMPT.format(
            query=query,
            source_summary=source_summary,
            reviewer_feedback=reviewer_feedback or "(no reviewer feedback yet)",
            already_run=already_run,
            count=count,
        )
        response = llm.invoke(prompt_text)
        text = response.content if hasattr(response, "content") else str(response)
    except Exception:  # noqa: BLE001
        # Fall back to template queries if LLM call fails.
        emit_progress(config, "Researcher: gap query LLM failed, using templates", kind="warning")
        return generate_gap_queries(query, intensity)

    # Parse: one query per line, strip list markers. Shared helper so the
    # flat and per-item generators accept the same LLM output shapes.
    queries = _parse_query_lines(text, limit=count)

    if not queries:
        emit_progress(
            config, "Researcher: gap query LLM returned no queries, using templates", kind="warning"
        )
        return generate_gap_queries(query, intensity)

    return queries


def _fresh(queries, executed_queries: set[str]) -> list[str]:
    """Deduplicate a query list against executed queries."""
    return [q for q in queries if q not in executed_queries]


def generate_gap_queries_for_items(
    query: str,
    intensity: int,
    items: list[dict],
    executed_queries: set[str],
    sources: list | None = None,
    config: RunnableConfig | None = None,
) -> list[str]:
    """Generate per-item gap queries for open review items.

    Falls back to generate_gap_queries_dynamic when there are no open
    blocking/required items (first-pass behavior), the LLM call fails, or
    the LLM returns no usable queries. sources (when given) give the
    dynamic fallback real source context instead of an empty summary.
    """
    sources = sources or []
    open_items = [i for i in items if i.get("status") == "open"]
    if not open_items or intensity < 3:
        return generate_gap_queries_dynamic(query, intensity, sources, "", executed_queries, config)

    count = {3: 6, 4: 9, 5: 12}.get(intensity, 6)
    if len(open_items) > 6:
        # The prompt only carries the first 6 items, so exhaustion decisions
        # over larger item sets are visible in the progress stream.
        emit_progress(
            config,
            f"Researcher: {len(open_items)} open review items, generating queries for the first 6",
            kind="warning",
        )
    item_lines = "\n".join(f"- [{i['category']}] {i['text']}" for i in open_items[:6])
    executed_sorted = sorted(executed_queries)
    already_run = "\n".join(f"- {q}" for q in executed_sorted[-30:]) or "(none yet)"

    try:
        settings = load_config()
        llm = get_llm(get_researcher_model(settings), temperature=0.8)
        response = llm.invoke(
            ITEM_GAP_QUERY_PROMPT.format(
                query=query,
                review_items=item_lines,
                already_run=already_run,
                count=count,
            )
        )
        text = response.content if hasattr(response, "content") else str(response)
    except Exception:  # noqa: BLE001
        emit_progress(
            config, "Researcher: item gap query LLM failed, using defaults", kind="warning"
        )
        return generate_gap_queries_dynamic(query, intensity, sources, "", executed_queries, config)

    queries = []
    for parsed in _parse_query_lines(text, limit=count):
        if len(parsed) > 4 and parsed not in queries:
            queries.append(parsed)
    if not queries:
        emit_progress(
            config,
            "Researcher: item gap query LLM returned no usable queries, using defaults",
            kind="warning",
        )
        return generate_gap_queries_dynamic(query, intensity, sources, "", executed_queries, config)
    return _fresh(queries, executed_queries)


def _research_concurrency() -> int:
    """Bounded parallelism for scrape+extract. Env-tunable, minimum 1."""
    try:
        return max(1, int(os.environ.get("ORA_RESEARCH_CONCURRENCY", "4")))
    except ValueError:
        return 4


def _scrape_and_extract_one(
    url: str,
    normalized_url: str,
    title: str,
    max_content_chars: int,
    query: str,
    intensity: int,
    model_name: str,
    config: RunnableConfig | None,
) -> dict:
    """Scrape one URL and evaluate it. Safe to call from a worker thread.

    Never raises: all failures are reported in the returned dict so a single bad
    URL cannot abort the batch.
    """
    from ora.tools.evaluate import evaluate_source
    from ora.tools.extract import extract_and_evaluate
    from ora.tools.scrape import scrape_page

    display_url = url.replace("https://", "").replace("http://", "")[:80]
    out = {
        "url": url,
        "normalized_url": normalized_url,
        "ok": False,
        "source": None,
        "extraction": None,
        "claim_text": "",
        "log": [],
        "events": [],
    }
    try:
        content = scrape_page.invoke({"url": url})
    except Exception as e:  # noqa: BLE001
        out["log"].append(f"  Scraped: 0 chars from {url[:60]} (FAIL: {e})")
        out["events"].append((f"Researcher: scrape failed for {display_url}", "error"))
        return out

    is_error = content.startswith(("Scrape error", "Scrape failed", "No content extracted"))
    out["log"].append(
        f"  Scraped: {len(content)} chars from {url[:60]} {'(FAIL)' if is_error else ''}"
    )
    if is_error:
        out["events"].append((f"Researcher: scrape failed for {display_url}", "error"))
        return out

    out["events"].append(
        (f"Researcher: scraped {len(content)} chars from {display_url}", "success")
    )
    content = content[:max_content_chars]

    try:
        if intensity >= 3:
            source, extraction = extract_and_evaluate(
                url=url,
                title=title,
                content=content,
                source_type="unknown",
                query=query,
                config=config,
                max_chars=max_content_chars,
                model_name=model_name,
            )
            claim_text = extraction.summary if extraction.summary else content[:500]
            out["log"].append(
                f"  Extracted: {len(extraction.key_claims)} claims, "
                f"{len(extraction.recommendations)} recommendations, "
                f"reliability={extraction.source_reliability}"
            )
        else:
            source = evaluate_source(
                url=url, title=title, content=content, source_type="unknown"
            )
            extraction = None
            claim_text = content[:500]
    except Exception as e:  # noqa: BLE001
        out["log"].append(f"  Source eval failed from {url[:60]}: {e}")
        out["events"].append(
            (f"Researcher: source evaluation failed for {display_url}", "error")
        )
        source = Source(
            url=url,
            title="",
            source_type="unknown",
            overall_reliability="Low",
            notes=f"Source evaluation failed: {e}",
        )
        extraction = None
        claim_text = content[:500] if content else ""
    else:
        out["events"].append(("Researcher: evaluated source reliability", "info"))

    out.update(ok=True, source=source, extraction=extraction, claim_text=claim_text)
    return out


def _scrape_and_collect(
    urls: list[str],
    params: dict,
    max_content_chars: int,
    config: RunnableConfig | None,
    log: list[str],
    sources: list,
    findings: list,
    seen_urls: set[str],
    url_titles: dict[str, str],
    *_,
    min_sources: int,
    query: str = "",
    intensity: int = 2,
    model_name: str = "",
    force_scrape: bool = False,
) -> bool:
    """Scrape URLs up to the per-query cap, appending to sources/findings.

    Returns True if we've hit the overall min_sources target and the caller
    should stop further work.
    """
    scraped_this_query = 0
    pending: list[str] = []
    pending_keys: set[str] = set()
    for url in urls[: params["urls_per_query"]]:
        if len(sources) >= min_sources and not force_scrape:
            return True

        normalized_url = _normalize_url_for_dedupe(url)
        if normalized_url in seen_urls or normalized_url in pending_keys:
            display_url = url.replace("https://", "").replace("http://", "")[:80]
            log.append(f"  Skipping duplicate source URL: {url[:80]}")
            emit_progress(config, f"Researcher: skipping duplicate {display_url}", kind="info")
            continue

        if _should_skip_url(url):
            display_url = url.replace("https://", "").replace("http://", "")[:80]
            log.append(f"  Skipping known-hostile domain: {url[:80]}")
            emit_progress(
                config, f"Researcher: skipping {display_url} (hostile domain)", kind="info"
            )
            continue

        pending.append(url)
        pending_keys.add(normalized_url)

    width = min(_research_concurrency(), params.get("scrapes_per_query", 4) or 4)
    for start in range(0, len(pending), width):
        if len(sources) >= min_sources and not force_scrape:
            return True
        if scraped_this_query >= params["scrapes_per_query"]:
            # Cap already reached in a prior chunk: do not submit another
            # batch of scrapes/extractions.
            break

        chunk = pending[start : start + width]
        emit_progress(
            config,
            f"Researcher: scraping {len(chunk)} pages concurrently",
            kind="scrape",
        )
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=len(chunk)) as pool:
            futures = {
                pool.submit(
                    _scrape_and_extract_one,
                    url,
                    _normalize_url_for_dedupe(url),
                    url_titles.get(_normalize_url_for_dedupe(url), ""),
                    max_content_chars,
                    query,
                    intensity,
                    model_name,
                    config,
                ): url
                for url in chunk
            }
            for fut in as_completed(futures):
                results.append(fut.result())

        order = {u: i for i, u in enumerate(chunk)}
        results.sort(key=lambda r: order.get(r["url"], 0))

        for r in results:
            log.extend(r["log"])
            for message, kind in r["events"]:
                emit_progress(config, message, kind=kind)
            if not r["ok"]:
                continue
            if len(sources) >= min_sources and not force_scrape:
                return True

            source, extraction, claim_text = r["source"], r["extraction"], r["claim_text"]
            finding_confidence = {
                "High": "High",
                "Medium": "Moderate",
                "Low": "Low",
            }.get(source.overall_reliability, "Unknown")
            sources.append(source)
            seen_urls.add(r["normalized_url"])
            findings.append(
                Finding(
                    claim=claim_text or "",
                    confidence=finding_confidence,  # type: ignore[arg-type]
                    supporting_sources=[r["url"]],
                    extraction=extraction,
                )
            )
            scraped_this_query += 1
            if scraped_this_query >= params["scrapes_per_query"]:
                break

    return len(sources) >= min_sources


def researcher_node(state: ResearchState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Researcher: iterate until min_sources target is reached.

    Synchronous node -- uses blocking HTTP calls inside LangGraph's sync execution.
    Queries already executed across *all* researcher invocations are deduplicated
    so the reviewer sending execution back to researcher does not waste search calls.
    Gap queries after round 1 are generated dynamically by the LLM using context
    about found sources and reviewer feedback.
    """
    settings = load_config()
    model_name = get_researcher_model(settings)

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
        _normalize_url_for_dedupe(source.url) for source in sources if hasattr(source, "url")
    }
    log: list[str] = []

    # Dedup set: queries executed across ALL researcher invocations.
    # This prevents wasted search calls when the reviewer sends execution back.
    executed_q_set: set[str] = set(state.get("executed_queries", []))
    if executed_q_set:
        log.append(f"Resuming with {len(executed_q_set)} previously-executed queries deduped")

    # Reviewer feedback for targeted gap queries.
    reviewer_feedback = _format_reviewer_feedback(state)
    revise_round = bool(reviewer_feedback)
    # Pass-level marker: revise_round is cleared after the first loop round
    # (it only forces one extra evidence round when min_sources is already
    # met), but the post-loop exhaustion rule below needs to know the whole
    # pass was a review-driven revise.
    revise_pass = bool(reviewer_feedback)

    # Structured open review items from the last reviewer verdict. When any
    # exist, gap queries target each item instead of the flat dynamic
    # generator; items stay "open" until this pass either finds evidence for
    # them or exhausts them below.
    open_items = [i for i in state.get("review_items") or [] if i.get("status") == "open"]

    def round_gap_queries() -> list[str]:
        """Gap-query source for the current round, shared by the round-gap
        branch and the duplicate-regeneration branch so both call identical
        fallback logic: per-item generation when structured open review items
        exist (its own fallbacks carry the same sources context), otherwise
        the flat dynamic generator with prose reviewer feedback."""
        if open_items:
            return generate_gap_queries_for_items(
                query=query,
                intensity=intensity,
                items=open_items,
                executed_queries=executed_q_set,
                sources=sources,
                config=config,
            )
        return generate_gap_queries_dynamic(
            query=query,
            intensity=intensity,
            sources=sources,
            reviewer_feedback=reviewer_feedback,
            executed_queries=executed_q_set,
            config=config,
        )

    # Baseline for this pass's deltas. sources/findings are appended to in
    # place by _scrape_and_collect inside the loop, so lengths captured here
    # (before the loop) are the correct baseline for the counts returned as
    # last_round_new_sources/last_round_new_findings.
    start_sources = len(sources)
    start_findings = len(findings)

    # Queries actually searched this pass (not merely generated); a failed
    # search still counts as an attempt. The exhaustion rule below requires
    # >= 2 attempts before marking open items evidence_exhausted, so items
    # are never exhausted on a single unlucky search.
    pass_executed_queries: list[str] = []

    round_num = 0

    while (len(sources) < min_sources or revise_round) and round_num < max_rounds:
        round_num += 1
        emit_progress(
            config,
            f"Researcher: round {round_num} (have {len(sources)}, need {min_sources})",
            kind="info",
        )

        # True on the forced revise round of a pass that already met
        # min_sources with open review items: queries must be item-targeted
        # and up to three of them searched before the round ends (see below).
        item_targeted_revise = bool(revise_round and open_items and len(sources) >= min_sources)

        if round_num == 1:
            if item_targeted_revise:
                # On this round, skip leftover plan search_queries: they are
                # untargeted first-pass queries that the researcher never
                # consumed, and zero-yield searches on them would trip the
                # whole-pass exhaustion rule below without the per-item gap
                # generator ever running. Go straight to the item-targeted
                # path. Plan queries remain the round-1 source for first
                # passes and for revise passes still chasing min_sources.
                queries_for_round = round_gap_queries()
            else:
                plan_queries = state.get("search_queries", [])
                if plan_queries:
                    queries_for_round = plan_queries
                else:
                    queries_for_round = list(generate_search_queries(query, intensity))
        else:
            # Gap queries: adapt to what's been found and what the reviewer
            # flagged. Falls back to templates on failure. With structured
            # open review items, target each item instead (see round_gap_queries).
            queries_for_round = round_gap_queries()

        # Filter out queries already executed in any prior invocation.
        fresh_queries = [q for q in queries_for_round if q not in executed_q_set]
        if not fresh_queries and (round_num > 1 or revise_round):
            # All gap queries are duplicates -- try one more LLM generation
            # with explicit instruction to avoid repeats.
            emit_progress(
                config,
                "Researcher: all gap queries were duplicates, regenerating...",
                kind="warning",
            )
            queries_for_round = round_gap_queries()
            fresh_queries = [q for q in queries_for_round if q not in executed_q_set]
            if not fresh_queries:
                if revise_round and open_items:
                    # A revise pass with open items and nothing new to try:
                    # stop instead of falling back to generic templates so the
                    # writer/reviewer can close the loop (the whole pass is
                    # judged by the exhaustion rule after the loop).
                    emit_progress(
                        config,
                        "Researcher: no fresh queries for open review items, ending revise pass",
                        kind="warning",
                    )
                    break
                # LLM regeneration produced only duplicates -- fall back
                # to template-based gap queries as a last resort.
                queries_for_round = generate_gap_queries(query, intensity)
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

        # On an item-targeted revise round (min_sources already met), do NOT
        # end the round after the first query: _scrape_and_collect returns
        # True immediately at min_sources, which would otherwise give the
        # open items only a single-query "attempt" per pass. Search up to
        # three fresh queries instead so a genuine multi-query attempt happens
        # within one pass (the exhaustion rule below requires >= 2 executed
        # searches). Rounds still chasing min_sources keep the old behavior
        # of searching every fresh query.
        round_queries = fresh_queries if not item_targeted_revise else fresh_queries[:3]
        for q in round_queries:
            if len(sources) >= min_sources and not revise_round:
                break

            executed_q_set.add(q)
            # Count only queries that are actually searched this pass. A
            # failed search still counts: it is a genuine (if unproductive)
            # attempt toward the open items.
            pass_executed_queries.append(q)
            log.append(f"Search: {q}")
            emit_progress(config, f'Researcher: searching "{q}"', kind="search")
            r = web_search.invoke({"query": q})
            log.append(f"  Result: {len(r)} chars")
            search_failed = r.startswith(("Search error", "Search failed"))
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

            scrape_done = _scrape_and_collect(
                urls,
                params,
                max_content_chars,
                config,
                log,
                sources,
                findings,
                seen_urls,
                url_titles,
                min_sources=min_sources,
                query=query,
                intensity=intensity,
                model_name=model_name,
                force_scrape=revise_round,
            )
            if scrape_done and not item_targeted_revise:
                # force_scrape suppresses _scrape_and_collect's internal
                # early-return at min_sources, so all eligible URLs for this
                # query are scraped. Its True return still ends the query
                # loop on ordinary rounds; item-targeted revise rounds keep
                # searching up to the 3-query cap so open items get a
                # genuine attempt.
                break

        if revise_round:
            revise_round = False

    # Exhaustion rule: mark open items evidence_exhausted only when this was
    # a review-driven revise pass, open items exist, the WHOLE pass produced
    # zero new sources, AND at least two queries were actually searched. On an
    # item-targeted revise round where min_sources is already met the query
    # loop now runs up to three searches (item_targeted_revise above), so two
    # zero-yield searches are a genuine multi-query attempt; one search (or
    # zero after regeneration) leaves items open so attempts can accumulate
    # across
    # REVISE passes. NOTE for the amended Task 4 routing contract: REVISE
    # routes back to the researcher iff revision_count < 3 AND at least one
    # open (non-exhausted) review item remains, otherwise the graph ends;
    # zero-progress passes still route back while an open item remains.
    # NOTE for the Task 4 reviewer re-audit: review_items_from_verdict maps
    # any repeated gap back to status "open", so the reviewer must treat
    # items the researcher already marked evidence_exhausted distinctly (an
    # acceptable documented gap) instead of REVISE-ing them anew.
    # Attribution is intentionally coarse (whole-pass, not per-item): gap
    # queries are generated per item, but scraped evidence is collected into
    # one shared pool, so a source cannot be reliably assigned to the item
    # that motivated it.
    review_items = state.get("review_items") or []
    if (
        revise_pass
        and open_items
        and (len(sources) - start_sources) == 0
        and len(pass_executed_queries) >= 2
    ):
        for item in review_items:
            if item.get("status") == "open":
                item["status"] = "evidence_exhausted"

    if not findings:
        results_text = web_search.invoke({"query": query})
        findings.append(
            Finding(
                claim=f"No scraped content found. Raw search: {results_text[:300]}",
                confidence="Unknown",
            )
        )

    research_status = "final" if len(sources) >= min_sources else "interim"

    source_label = "source" if len(sources) == 1 else "sources"
    finding_label = "finding" if len(findings) == 1 else "findings"
    emit_progress(
        config,
        f"Researcher: finished with {len(sources)} {source_label} and {len(findings)} {finding_label} ({research_status})",
        kind="success",
    )

    return {
        "executed_queries": list(executed_q_set),
        "sources": sources[prior_source_count:],
        "findings": findings[prior_finding_count:],
        "research_status": research_status,
        "messages": ["\n".join(log)],
        "review_items": review_items,
        "last_round_new_sources": len(sources) - start_sources,
        "last_round_new_findings": len(findings) - start_findings,
    }
