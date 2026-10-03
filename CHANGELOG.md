# Changelog

## Unreleased

Search reliability and research throughput.

- New `decodo` search provider (`search.provider: decodo`), using Decodo's
  Google SERP API. Credentials come from `DECODO_USERNAME` / `DECODO_PASSWORD`.
  Search failures are reported as errors rather than silently producing an
  empty result; the Firecrawl fallback is off by default and must be opted into
  with `search.fallback_to_firecrawl: true`.
- Firecrawl search honors the `FIRECRAWL_API_URL` env override again, matching
  scrape behavior and `CONFIG.md`.
- Research now scrapes and extracts each query's URLs concurrently
  (`ORA_RESEARCH_CONCURRENCY`, default 4). An intensity-4 run that previously
  failed to finish in 45 minutes now completes in about 12 minutes with more
  sources. Depth per intensity level is unchanged.
- New `llm_timeout_seconds` (default 300) and `llm_max_retries` (default 2)
  settings, settable in `config.yaml` or via `ORA_LLM_TIMEOUT_SECONDS` /
  `ORA_LLM_MAX_RETRIES`. These bound each LLM call; previously the SDK defaults
  applied (600s read, 2 retries), so a single stalled call could block a run for
  about 30 minutes. See `CONFIG.md`.
- Usage and cost accounting now works from concurrent workers.

## 0.1.0

Initial public release of ORA.

- Multi-agent research CLI.
- PyPI package name `open-research-agent`, with `open-research-agent` as the primary CLI command and `ora` as an alias.
- Research planning flow.
- Configurable intensity levels 1-5.
- Firecrawl search and scrape integration.
- DeepSeek-backed planning, research, writing, and review agents.
- Optional adversarial review for higher-intensity research.
- Local markdown report output.
