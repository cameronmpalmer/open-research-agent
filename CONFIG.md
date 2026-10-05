# ORA Configuration Reference

ORA reads its settings from `~/.ora/config.yaml`. Generate a starter file with
`open-research-agent config --init`, view the effective configuration with
`open-research-agent config --show`, and see every CLI flag with
`open-research-agent --help`.

This document is the complete reference for the YAML format, the environment
variables ORA understands, and how they interact.

## Precedence

Settings are resolved in this order:

1. **YAML file** (`~/.ora/config.yaml`, or `ORA_`-prefixed env vars applied by
   the settings layer first, then YAML on top)
2. **Environment variables** (fields not present in the YAML file)
3. **Defaults** (hardcoded in `ora/config.py`)

In practice this means: **the YAML file wins over environment variables for
any key it defines**. An env var is only honored when the YAML file does not
set that key. (The one exception is API keys, which are read directly from the
process environment; see [API keys](#api-keys).)

## Full example

This is everything `config --init` writes, plus the optional `output:` block
and legacy key, annotated:

```yaml
# ~/.ora/config.yaml

models:
  default: deepseek-v4-flash       # every role falls back to this
  # Optional per-role overrides; any unset role uses models.default
  # (an unset writer instead follows models.researcher):
  # researcher: <model>            # researcher agent, query generation, extractor
  # supervisor: <model>            # planning; no CLI flag exists for this
  # writer: <model>                # report synthesis
  # reviewer: <model>              # adversarial review (intensity 3+)

search:
  provider: firecrawl
  firecrawl_api_url: https://api.firecrawl.com   # set to http://localhost:3002 for self-hosted

output:                              # parsed but not yet consumed by the CLI
  default_format: markdown
  always_include_sources: true

limits:                              # written by config --init and shown by
  max_revisions: 3                   # config --show; the revision budget when
  default_intensity: 2               # --max-revisions is not passed

provider:
  default: deepseek                  # used when a model name has no prefix

providers:
  deepseek:
    base_url: https://api.deepseek.com
    api_key_env: DEEPSEEK_API_KEY
  openrouter:
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    headers:                         # optional; sent on every request
      HTTP-Referer: https://github.com/cameronmpalmer/open-research-agent
      X-Title: ORA

# llm_timeout_seconds: 300.0   # optional; per-call read timeout (SDK default 600s)
# llm_max_retries: 2           # optional; retries per LLM call

# deepseek_base_url: https://api.deepseek.com   # legacy key, see below
```

## Sections

### models

Which model each agent role uses. **`models.default` applies to every role**:
an unset `researcher`, `supervisor`, or `reviewer` falls back to `default`, and
an unset `writer` falls back to the resolved researcher model.
Model names may carry a `provider:model` prefix (see
[Provider routing](#provider-routing)).

| Key | Default | Used by |
|---|---|---|
| `models.default` | `deepseek-v4-flash` | Every role that has no explicit override |
| `models.researcher` | (falls back to `models.default`) | Researcher agent, query generation, per-source extractor |
| `models.supervisor` | (falls back to `models.default`) | Supervisor (planning and routing); config only, no CLI flag |
| `models.writer` | (falls back to `models.researcher`) | Writer (report synthesis) |
| `models.reviewer` | (falls back to `models.default`) | Reviewer (intensity 3+); overridable with `--reviewer-model` |

CLI overrides: `--model` sets the researcher model for one run, and the writer
follows it unless `models.writer` is set; `--reviewer-model` sets the reviewer
model for one run. Overrides are run-scoped and applied at the config layer, so
they cover every internal call (including gap-query generation and per-source
extraction); no call site bypasses the configured model. `ora research` prints
the resolved model for all four roles before generating the plan.

### search

| Key | Default | Purpose |
|---|---|---|
| `search.provider` | `firecrawl` | Search backend: `firecrawl` or `decodo`. Unrecognized values warn and fall back to `firecrawl`. |
| `search.firecrawl_api_key` | (unset) | Firecrawl key; normally exported as `FIRECRAWL_API_KEY` instead |
| `search.firecrawl_api_url` | `https://api.firecrawl.com` | Firecrawl endpoint; set to `http://localhost:3002` for self-hosted Firecrawl (key optional there) |
| `search.decodo_api_url` | `https://scraper-api.decodo.com/v2/scrape` | Decodo SERP endpoint (`target: google_search`) |
| `search.decodo_api_key_env` | `DECODO_API_KEY` | Name of the env var holding the Decodo Basic-auth API key |
| `search.decodo_domain` | `com` | Google domain to search |
| `search.decodo_locale` | `en-us` | Result locale |
| `search.fallback_to_firecrawl` | `false` | Opt in to retrying a Decodo *failure* via Firecrawl |

The Decodo credential is read from process env only and is never written to
`config.yaml`; the `*_env` setting names the variable to read. Enable Decodo
with `search.provider: decodo` plus `DECODO_API_KEY` exported. That key is the
Basic-auth credential from the Decodo dashboard (API Playground), used verbatim
as `Authorization: Basic <key>`. Username and password are deliberately **not**
supported: the key is the credential Decodo issues, and storing the account
password would expose a broader secret than the API needs. Basic auth base64 is
encoding, not encryption, so treat the key as a bearer secret and rotate it
from the dashboard. `ora research` checks the key before doing any work: it
exits with an error rather than generating a plan when the key is missing or
when Decodo rejects it. That check is a single probe request against an invalid
target, which validates the credential without scraping anything.

The fallback is deliberate and narrow, and it is **off by default**: Firecrawl
search is unreliable and in practice often returns nothing, so silently falling
back would hide Decodo failures. Set `search.fallback_to_firecrawl: true` to
opt in. When enabled, a Decodo **failure** (bad credentials, HTTP error,
provider-reported error, or a malformed body) retries via Firecrawl; either way,
a successful search that simply found nothing is reported as
`No search results found.` and does **not** retry, so a quiet query is not
double-billed. Scrape always uses Firecrawl regardless of `search.provider`.

### output

Parsed and shown by `config --init`/`--show`, but **not yet consumed by the
CLI**. Reserved for future report-format controls.

| Key | Default |
|---|---|
| `output.default_format` | `markdown` |
| `output.always_include_sources` | `true` |

### limits

`limits.max_revisions` is now wired: it is the revision budget used whenever
`--max-revisions` is not passed explicitly (an explicit flag wins).
`limits.default_intensity` is parsed and shown by `config --show` but the CLI
still uses its own `--intensity` default of 2.

| Key | Default |
|---|---|
| `limits.max_revisions` | `3` |
| `limits.default_intensity` | `2` |

### provider

| Key | Default | Purpose |
|---|---|---|
| `provider.default` | `deepseek` | Provider used when a model name carries no `provider:` prefix |

### providers

A map of provider name to connection settings. `deepseek` and `openrouter`
are built in with the defaults shown in the example; entries in this map
override those defaults (e.g. a different `base_url` or `api_key_env`), and
extra entries define additional providers.

| Key | Default | Purpose |
|---|---|---|
| `<name>.base_url` | per provider | OpenAI-compatible endpoint |
| `<name>.api_key_env` | `DEEPSEEK_API_KEY` / `OPENROUTER_API_KEY` | Name of the env var holding the key |
| `<name>.headers` | (none) | Extra HTTP headers, e.g. OpenRouter's `HTTP-Referer` / `X-Title` |

A provider without a `base_url` is rejected with a clear error at call time.
A model prefix naming a provider that is neither built in nor configured
warns and falls back to the default provider.

### LLM call bounds

Top-level scalars that bound each LLM request. They apply to every provider
and every agent call, so a stalled request cannot block a run indefinitely.

| Key | Default | Purpose |
|---|---|---|
| `llm_timeout_seconds` | `300.0` | Read timeout for one LLM call. The OpenAI SDK default is 600s; 300s leaves headroom over the slowest observed legitimate call (~155s) while capping the worst case. |
| `llm_max_retries` | `2` | Retries per LLM call after the first attempt. With the 300s timeout this bounds one logical call to roughly 15 minutes instead of the ~30 minutes the SDK defaults allow. |

Both are honored from YAML and from their `ORA_` env-var forms
(`ORA_LLM_TIMEOUT_SECONDS`, `ORA_LLM_MAX_RETRIES`); as everywhere else, a YAML
key wins over its env var.

### Legacy: `deepseek_base_url`

Before the `providers:` map existed, the DeepSeek base URL was a flat
top-level key. It is still honored:

- When the YAML has no `providers:` section, `deepseek_base_url` becomes the
  deepseek provider's base URL.
- When a `providers:` section exists, it wins over the legacy key.

## Environment variables

### API keys

API keys are read **directly from the process environment** at call time.
They are not read from `~/.ora/config.yaml` (the config only names which env
var to use) and are not loaded from a `.env` file.

| Variable | Used for |
|---|---|
| `DEEPSEEK_API_KEY` | deepseek provider (default) |
| `OPENROUTER_API_KEY` | openrouter provider |
| `OPENAI_API_KEY` | Legacy fallback for deepseek when `DEEPSEEK_API_KEY` is unset |
| `FIRECRAWL_API_KEY` | Search/scrape; overrides `search.firecrawl_api_key` |
| `FIRECRAWL_API_URL` | Search/scrape endpoint; overrides `search.firecrawl_api_url` |
| `DECODO_API_KEY` | Decodo Basic-auth API key (needed when `search.provider: decodo`) |
| `ORA_RESEARCH_CONCURRENCY` | Parallel scrape+extract workers per query batch. Default `4`, clamped to a minimum of `1`, and capped by `scrapes_per_query`. Higher values finish research faster at the cost of more simultaneous Firecrawl and LLM requests. Not a `config.yaml` key. |

### ORA_ settings variables

Every YAML key also has an env-var form using the `ORA_` prefix and `__` as
the nested separator, e.g. `ORA_MODELS__SUPERVISOR=deepseek-v4-pro`. These are
loaded by the settings layer from the environment **and from a `.env` file in
the working directory** (`env_file=".env"`), then the YAML file is applied on
top, so a YAML key beats its env var.

Common examples:

| Variable | Equivalent YAML |
|---|---|
| `ORA_MODELS__SUPERVISOR` | `models.supervisor` |
| `ORA_MODELS__WRITER` | `models.writer` |
| `ORA_MODELS__DEFAULT` | `models.default` |
| `ORA_PROVIDER__DEFAULT` | `provider.default` |
| `ORA_SEARCH__PROVIDER` | `search.provider` |
| `ORA_SEARCH__FIRECRAWL_API_URL` | `search.firecrawl_api_url` |
| `ORA_SEARCH__DECODO_API_URL` | `search.decodo_api_url` |
| `ORA_SEARCH__FALLBACK_TO_FIRECRAWL` | `search.fallback_to_firecrawl` |
| `ORA_DEEPSEEK_BASE_URL` | legacy `deepseek_base_url` |
| `ORA_LLM_TIMEOUT_SECONDS` | `llm_timeout_seconds` |
| `ORA_LLM_MAX_RETRIES` | `llm_max_retries` |
| `ORA_LIMITS__MAX_REVISIONS` | `limits.max_revisions` |

## Provider routing

A model name may carry a `provider:model` prefix:

```bash
open-research-agent research "..." --model openrouter:anthropic/claude-3.5-sonnet
```

- `openrouter:anthropic/claude-3.5-sonnet` routes to the `openrouter` provider
  with model `anthropic/claude-3.5-sonnet` (the split is on the first colon;
  slashes and any later colons stay in the model name).
- No prefix means the `provider.default` provider.
- An unknown provider prefix (e.g. `openai:gpt-4.1`) warns and falls back to
  the default provider, keeping the full name.
- Each agent role can use a different provider by setting its `models.*`
  entry with a prefix (supervisor and writer via config only).

## Related

- `open-research-agent config --init` writes the starter file.
- `open-research-agent config --show` prints the effective configuration,
  including the resolved provider list and whether each provider's key is set.
- `open-research-agent --help` documents every CLI flag.
