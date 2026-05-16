# Open Research Agent (ORA)

Open Research Agent (ORA) is an open-source multi-agent research CLI. ORA plans research, searches and scrapes web sources, synthesizes findings, and optionally uses an adversarial reviewer for higher-intensity research.

Current release: **0.1.0**

## What ORA does

ORA turns a research question into a sourced markdown report:

1. A supervisor drafts a research plan.
2. The researcher searches and scrapes web sources.
3. The writer synthesizes findings into a report.
4. For intensity levels 3 and above, an adversarial reviewer audits the draft.

## Current backend support

ORA 0.1.0 currently supports one LLM backend:

- **LLM backend:** DeepSeek API
- **Search and scraping backend:** Firecrawl

The default model names are `deepseek-v4-flash` for research and writing, and `deepseek-v4-pro` for planning and review. Other LLM backends are not currently supported.

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
| `--max-revisions N` | Cap reviewer revision rounds (default: 3) |
| `--quiet` | Suppress progress output, show only the final report |

## Output files

By default, `open-research-agent research` saves a timestamped markdown report in the current directory. Generated research reports are local outputs and should not be committed to the repository.

Use `--output` to choose a specific path, or `--no-save` to print the report without writing a file.

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run the CLI locally:

```bash
open-research-agent --help
ora --help
python -m ora --help
```

See `CONTRIBUTING.md` for contributor setup and repository hygiene expectations.

## License

MIT License. See `LICENSE`.
