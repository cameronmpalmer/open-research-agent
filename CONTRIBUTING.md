# Contributing

Thanks for your interest in improving Open Research Agent (ORA).

## Development setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configuration for local runs

ORA 0.1.0 currently supports the DeepSeek API as its LLM backend and Firecrawl for search and scraping.

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export FIRECRAWL_API_KEY="your-firecrawl-api-key"
```

You can create a local config file with:

```bash
ora config --init
```

## Run tests

```bash
pytest
```

For CLI-focused changes:

```bash
python3 -m pytest tests/test_cli.py -q
```

## Run the CLI locally

```bash
ora --help
python -m ora --help
ora config --show
```

## Generated reports

`ora research` can generate timestamped markdown reports in the repository root or current working directory. Do not commit generated research reports.

Use `--no-save` while testing if you do not need a report file:

```bash
ora research "AI memory systems" --no-save
```

## Secrets

Do not commit API keys, `.env` files, credentials, or generated files containing private research output.

## Pull requests

Before opening a pull request:

1. Run the relevant focused tests.
2. Run the full test suite with `pytest`.
3. Check `git status --short` for generated reports or other accidental files.
4. Update documentation when behavior changes.
