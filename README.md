# Open Research Agent (ORA)

An open-source multi-agent research tool with a built-in adversarial reviewer. Built with LangChain and LangGraph.

**ORA** = Open Research Agent. Submit a query with an intensity level, get back calibrated, source-traced research that has been audited by a separate adversarial agent.

## Status

Pre-alpha. Spec and design in progress.

## Key Differentiators

- **Adversarial reviewer** -- separate agent with fresh context attacks findings, verifies URLs, searches for contradicting evidence
- **Calibrated confidence** -- every claim has an explicit confidence level (High / Moderate / Low / Unknown)
- **Source traceability** -- claims traced back to root sources with corroboration tracking
- **Intensity levels** -- configurable depth from quick check to exhaustive research
- **Benchmark-validated** -- scored against the Deep Research Bench
