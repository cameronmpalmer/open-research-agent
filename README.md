# Open Research Audit (ORA)

An open-source multi-agent research tool that audits its own findings. Built with LangChain and LangGraph.

**ORA** = Open Research Audit. Submit a query with an intensity level, get back calibrated, source-traced research with a built-in adversarial reviewer.

## Status

Pre-alpha. Spec and design in progress.

## Key Differentiators

- **Adversarial reviewer** -- separate agent with fresh context attacks findings, verifies URLs, searches for contradicting evidence
- **Calibrated confidence** -- every claim has an explicit confidence level (High / Moderate / Low / Unknown)
- **Source traceability** -- claims traced back to root sources with corroboration tracking
- **Intensity levels** -- configurable depth from quick check to exhaustive research
- **Benchmark-validated** -- scored against the Deep Research Bench
