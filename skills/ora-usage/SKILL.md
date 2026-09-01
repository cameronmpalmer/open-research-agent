---
name: ora-usage
description: Use when running ORA (Open Research Agent) CLI commands, configuring ORA, understanding intensity levels, troubleshooting ORA errors, or when asked to research a topic with ORA. Triggers on "ora research", "ora plan", "ora config", "open-research-agent", ORA output files, ORA configuration, or ORA error messages.
---

# ORA Usage

## Overview

ORA is a multi-agent research CLI (v0.1.0) that turns a query into a sourced markdown report. Pipeline: Supervisor plans → Researcher searches and scrapes → Writer synthesizes → Reviewer audits (intensity 3+). Backends: DeepSeek API (LLM), Firecrawl (search/scrape).

## Required Setup

```bash
export DEEPSEEK_API_KEY="your-key"    # or OPENAI_API_KEY
export FIRECRAWL_API_KEY="your-key"   # omit for self-hosted Firecrawl
export OPENROUTER_API_KEY="your-key"    # only if using OpenRouter models
ora config --init                     # creates ~/.ora/config.yaml
```

For self-hosted Firecrawl: set `FIRECRAWL_API_URL=http://localhost:3002`.

## Quick Reference

| Task | Command |
|------|---------|
| Run research | `ora research "query"` (default intensity 2) |
| Run at higher intensity | `ora research "query" --intensity 4` |
| Preview plan only | `ora plan "query"` |
| Custom output path | `ora research "query" --output path/to/report.md` |
| Stdout only, no file | `ora research "query" --no-save` |
| Skip interactive approval | `ora research "query" -y` (`--auto-approve`) |
| Minimal output | `ora research "query" --quiet` |
| Change researcher model | `ora research "query" --model deepseek-v4-chat` |
| Change reviewer model | `ora research "query" --reviewer-model deepseek-v4-pro` |
| Limit reviewer rounds | `ora research "query" --max-revisions 2` |
| Show config | `ora config --show` |

## Configuration

Config file: `~/.ora/config.yaml`. **Priority: env vars > config file > defaults.**

```yaml
models:
  default: deepseek-v4-flash       # researcher + writer fallback
  researcher: ~                     # overrides default
  supervisor: deepseek-v4-pro       # planning (no CLI flag exists)
  reviewer: deepseek-v4-pro
search:
  provider: firecrawl
limits:
  max_revisions: 3
  default_intensity: 2
```

Change supervisor model only via config file or `ORA_MODELS__SUPERVISOR` env var.

`provider:model` prefixes are functional: `openrouter:anthropic/claude-3.5-sonnet`
routes to OpenRouter, `deepseek:deepseek-chat` to DeepSeek. No prefix uses the
default provider (`provider.default` in config, default `deepseek`). Provider
base URLs and API-key env vars live under `providers:` in `config.yaml`
(`config --init` writes the block). Unknown prefixes warn and fall back to the
default provider.

## Intensity Levels

| Level | Label | Min Sources | Max Rounds | Reviewer | Search Angles |
|-------|-------|-------------|------------|----------|---------------|
| 1 | Quick | 3 | 5 | No | 1 |
| 2 | Standard | 8 | 5 | No | 3 |
| 3 | Thorough | 15 | 7 | Yes | 7 |
| 4 | Deep | 50 | 10 | Yes | 12 |
| 5 | Exhaustive | 100 | 10 | Yes | 16 |

Reviewer (levels 3+) audits draft and can issue REVISE to restart research (max 3 cycles; cap with `--max-revisions`). No reviewer at levels 1-2.

**Default safe choice:** Level 2 for most research. Level 3+ when you need adversarial audit-trail review.

## Common Patterns

- **Quick fact-check:** Intensity 1-2 (no reviewer, 3-8 sources)
- **Preview before committing:** `ora plan "query"` or cancel at interactive prompt
- **Batch/scripted:** Use `-y` (auto-approve) + `--quiet` + `--output`
- **Model per phase:** `--model` = researcher + writer; `--reviewer-model` = reviewer; supervisor via config only

## Known Issues (v0.1.0)

- **`--no-review` is dead code.** Declared but not wired. Only dropping to intensity 2 removes the reviewer.

## Common Mistakes

| Mistake | Reality |
|---------|---------|
| Using `--no-review` at intensity 3+ | Flag is dead. Drop to intensity 2 instead. |
| Passing `openai:gpt-4.1` as model name | `openai` is not a configured provider, so it warns and falls back to the default provider. Use `openrouter:...` to route via OpenRouter. |
| Expecting `--supervisor-model` flag | Does not exist. Use config file or `ORA_MODELS__SUPERVISOR`. |
| Forgetting `FIRECRAWL_API_KEY` | Required for search/scrape. Report generation fails without it. |
| `--max-revisions` at intensity 1-2 | Only relevant at intensity 3+ (reviewer active). |
| Confused by auto-generated filename | Default: `{query-slug}-{timestamp}.md`. Use `--output` to control. |

## Red Flags

- "I'll use --no-review" → Dead code. Drop intensity.
- "I'll pass openai:gpt-4" → `openai` is an unknown provider; it warns and falls back to the default. Use `openrouter:...` or configure a provider.
- "I'll set --supervisor-model" → Flag does not exist. Use config.
- "Report didn't generate" → Check both `DEEPSEEK_API_KEY` and `FIRECRAWL_API_KEY`.
