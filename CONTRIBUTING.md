# Contributing

Thanks for your interest in improving Open Research Agent (ORA).

## First-time contributors

If you are new to contributing to open source, here is the standard workflow. ORA uses the **fork-and-PR model**: external contributors fork the repo, work on a branch, and open a pull request.

1. **Find an issue.** Look for issues labeled
   [`good first issue`](https://github.com/cameronmpalmer/open-research-agent/labels/good%20first%20issue).
   Comment on the issue to let others know you are working on it.

2. **Fork the repo.** Click **Fork** at the top-right of the
   [repository page](https://github.com/cameronmpalmer/open-research-agent).

3. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/open-research-agent.git
   cd open-research-agent
   ```

4. **Create a branch** with a short, descriptive name:
   ```bash
   git checkout -b feat/your-feature-name
   # or fix/your-bugfix-name, docs/what-you-updated, etc.
   ```

5. **Make your changes.** Follow the sections below for dev setup and testing.

6. **Open a pull request** from your fork's branch to `main` on the upstream repo.
   Reference the issue number in the PR description (e.g., `Closes #6`).
   Tag `@cameronmpalmer` as a reviewer.

7. **Iterate on review feedback.** Push additional commits to the same branch;
   they will appear in the open PR automatically.

## Development setup

Use Python 3.10.

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
open-research-agent config --init
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
open-research-agent --help
ora --help
python -m ora --help
open-research-agent config --show
```

## Generated reports

`open-research-agent research` can generate timestamped markdown reports in the repository root or current working directory. Do not commit generated research reports.

Use `--no-save` while testing if you do not need a report file:

```bash
open-research-agent research "AI memory systems" --no-save
```

## Secrets

Do not commit API keys, `.env` files, credentials, or generated files containing private research output.

## Pull requests

Before opening a pull request:

1. Run the relevant focused tests.
2. Run the full test suite with `pytest`.
3. Check `git status --short` for generated reports or other accidental files.
4. Update documentation when behavior changes.
