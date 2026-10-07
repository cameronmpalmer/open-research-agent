# Open Research Agent (ORA)

Open Research Agent (ORA) is an open-source multi-agent research CLI. ORA plans research, searches and scrapes web sources, synthesizes findings, and optionally uses an adversarial reviewer for higher-intensity research.

Current release: **0.2.0**

## What ORA does

ORA turns a research question into a sourced markdown report:

1. A supervisor drafts a research plan.
2. The researcher searches and scrapes web sources.
3. At intensity 3+, an LLM extraction layer pulls key claims, data, and entities from each source.
4. The writer synthesizes findings into a report.
5. For intensity levels 3 and above, an adversarial reviewer audits the draft.

## Current backend support

ORA 0.2.0 supports two LLM backends:

- **DeepSeek API** (default), models like `deepseek-v4-flash` and `deepseek-v4-pro`
- **OpenRouter**, an OpenAI-compatible gateway to many models, e.g. `anthropic/claude-3.5-sonnet` via `openrouter:anthropic/claude-3.5-sonnet`

Search and scraping use **Firecrawl**.

## Installation

Install from PyPI:

```bash
pip install open-research-agent
```

The primary CLI command is `open-research-agent`. The shorter `ora` command is also installed as a convenience alias.

Install from source for development:

```bash
git clone https://github.com/cameronmpalmer/open-research-agent.git
cd open-research-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configuration

Set the required API keys:

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export FIRECRAWL_API_KEY="your-firecrawl-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

Create a default config file:

```bash
open-research-agent config --init
```

Show the active configuration and intensity levels:

```bash
open-research-agent config --show
```

The config file is stored at:

```text
~/.ora/config.yaml
```

See [CONFIG.md](CONFIG.md) for the complete reference of the `config.yaml`
format, the available settings, and their environment-variable equivalents.

## Using a provider prefix

Any model name can carry a `provider:model` prefix to select the backend:

```bash
open-research-agent research "..." --model openrouter:anthropic/claude-3.5-sonnet
```

Without a prefix, calls use the default provider (`deepseek` by default). The
default can be changed in `config.yaml` under `provider.default`. An unknown
prefix (e.g. `openai:gpt-4.1`) warns and falls back to the default provider.
Each provider reads its API key from the environment variable named in its
`api_key_env`
(`DEEPSEEK_API_KEY` for deepseek, `OPENROUTER_API_KEY` for openrouter), and can
be configured under `providers:` in `config.yaml`:

```yaml
provider:
  default: deepseek
providers:
  deepseek:
    base_url: https://api.deepseek.com
    api_key_env: DEEPSEEK_API_KEY
  openrouter:
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    headers:
      HTTP-Referer: https://github.com/cameronmpalmer/open-research-agent
      X-Title: ORA
```

The legacy top-level `deepseek_base_url` key is still honored when no
`providers:` section exists.

## Quick start

Preview a research plan without running the full pipeline:

```bash
open-research-agent plan "What are the tradeoffs between Rust and Go for backend services?"
```

Run a standard research task:

```bash
open-research-agent research "What are the tradeoffs between Rust and Go for backend services?" --intensity 2
```

Run deeper research with adversarial review:

```bash
open-research-agent research "What are the tradeoffs between Rust and Go for backend services?" --intensity 4
```

Save to an explicit file:

```bash
open-research-agent research "AI memory systems" --output ai-memory-systems.md
```

Print only to stdout and do not save a report file:

```bash
open-research-agent research "AI memory systems" --no-save
```

## Intensity levels

ORA supports five research intensity levels:

| Level | Label | Minimum sources | Max rounds (safety cap) | Reviewer |
|---|---|---|---|---|
| 1 | Quick | 3 | 5 | No |
| 2 | Standard | 8 | 5 | No |
| 3 | Thorough | 15 | 7 | Yes |
| 4 | Deep | 50 | 10 | Yes |
| 5 | Exhaustive | 100 | 10 | Yes |

Levels 3, 4, and 5 use the adversarial reviewer by default.

## CLI flags

| Flag | Description |
|------|-------------|
| `-i`, `--intensity 1-5` | Research intensity level (default: 2) |
| `-o`, `--output PATH` | Save report to a specific path |
| `--no-save` | Print to stdout without saving a file |
| `--stdout` | Print to stdout (report is still saved) |
| `-m`, `--model NAME` | Override the LLM model for research and writing |
| `-r`, `--reviewer-model NAME` | Override the LLM model for planning and review |
| `-y`, `--auto-approve` | Skip the interactive plan approval prompt |
| `--no-review` | Disable adversarial reviewer (even at intensity 3+) |
| `--max-revisions N` | Maximum reviewer audits including the initial draft audit (`1` = single audit, no revision; defaults to `limits.max_revisions` in config, itself 3 = up to two revision passes) |
| `--quiet` | Suppress progress output, show only the final report |

## Output files

By default, `open-research-agent research` saves a timestamped markdown report in the current directory. Generated research reports are local outputs and should not be committed to the repository.

Use `--output` to choose a specific path, or `--no-save` to print the report without writing a file.

## Development

ORA requires **Python 3.10**. Later versions (3.11+) may encounter incompatibilities with the LangChain/LangGraph ecosystem.

The repository ships a `Makefile` that manages a local virtual environment automatically. From a clean clone:

```bash
make setup
```

Run the test suite (pass `T=tests/path` for a focused file, `ARGS="-x -q"` for pytest options):

```bash
make test
```

Lint and auto-format:

```bash
make lint
make format
```

Build the package (wheel and sdist into `dist/`):

```bash
make build
```

Run a research task from the Makefile. Reports are saved to `reports/` by default:

```bash
make research QUERY="What are the tradeoffs between Rust and Go for backend services?"
```

`make research QUERY="..." INTENSITY=4` runs at a higher intensity level; `OUTPUT=path.md` overrides the output path. `make plan QUERY="..."` previews a research plan without running it. `make check` runs lint then tests. See `make help` for the full target list.

Install the git pre-commit hook (run once per clone):

```bash
make install-hooks
```

The hook runs `make precommit` (lint, tests, and build) before every commit. Bypass it for a quick commit with `git commit --no-verify`.

Run the CLI locally:

```bash
open-research-agent --help
ora --help
python -m ora --help
```

See `CONTRIBUTING.md` for contributor setup and repository hygiene expectations.

## License

MIT License. See `LICENSE`.
