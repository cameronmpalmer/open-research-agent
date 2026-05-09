"""Click CLI for Open Research Agent (ORA)."""
import asyncio
import logging
import os
import warnings
import yaml
import click
from ora.config import load_config

logging.captureWarnings(True)
warnings.filterwarnings("ignore", message=".*allowed_objects.*")


def _run_async(coro):
    """Run an async function from sync Click command."""
    return asyncio.run(coro)


def _spin(coro, message="Working..."):
    """Run an async coroutine with a rich spinner."""
    from rich.console import Console
    console = Console()
    with console.status(message, spinner="dots"):
        return asyncio.run(coro)


def _print_markdown(text: str):
    """Render markdown to the terminal using rich."""
    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()
    console.print(Markdown(text))


@click.group()
@click.version_option(version="0.1.0", prog_name="ora")
def main():
    """Open Research Agent (ORA) - Multi-agent research with adversarial review.

    Submit a query, get back calibrated, source-traced research that
    has been audited by a separate adversarial agent.
    """
    pass


@main.command()
@click.argument("query")
@click.option("--intensity", "-i", type=click.IntRange(1, 5), default=2,
              help="Research intensity (1=Quick, 2=Standard, 3=Thorough)")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output file path (default: stdout)")
@click.option("--model", "-m", default=None,
              help="LLM model for researcher/writer (e.g., openai:gpt-4.1)")
@click.option("--reviewer-model", "-r", default=None,
              help="LLM model for adversarial reviewer")
@click.option("--max-revisions", type=int, default=3,
              help="Max writer-reviewer revision cycles")
@click.option("--no-review", is_flag=True,
              help="Skip adversarial review")
@click.option("--verbose", "-v", is_flag=True,
              help="Show agent actions in real-time")
def research(query, intensity, output, model, reviewer_model, max_revisions,
             no_review, verbose):
    """Run the full research pipeline."""
    if intensity > 3:
        click.echo("Levels 4-5 not yet implemented. Falling back to Level 3.", err=True)
        intensity = 3

    click.echo(f"ORA Research: {query}")
    click.echo(f"  Intensity: {intensity} | Model: {model or 'default'}")

    settings = load_config()
    if model:
        settings.models.researcher = model
    if reviewer_model:
        settings.models.reviewer = reviewer_model

    from ora.graph import build_graph
    graph = build_graph()
    config = {"configurable": {"thread_id": "research-1"}}

    initial_state = {
        "query": query,
        "intensity": intensity,
        "plan_approved": False,
        "revision_count": 0,
    }

    plan_result = _spin(graph.ainvoke(initial_state, config), message="Generating research plan...")
    plan = plan_result.get("research_plan", "No plan generated.")
    _print_markdown(plan)

    if not click.confirm("  Approve this plan and begin research?"):
        click.echo("  Research cancelled.")
        return

    plan_result["plan_approved"] = True
    plan_result["revision_count"] = plan_result.get("revision_count", 0)

    final_state = _spin(graph.ainvoke(plan_result, config), message="Researching...")

    revision_count = 0
    while revision_count < max_revisions:
        verdict = final_state.get("review_verdict")
        if verdict is None:
            break

        v = verdict.verdict if hasattr(verdict, 'verdict') else \
            (verdict.get("verdict", "PASS") if isinstance(verdict, dict) else "PASS")

        if v == "PASS":
            break

        click.echo(f"  Reviewer requested revision {revision_count + 1}/{max_revisions}")
        revision_count += 1
        final_state["revision_count"] = revision_count
        final_state = _spin(graph.ainvoke(final_state, config), message="Revising...")

    draft = final_state.get("draft_report", "No report generated.")
    verdict = final_state.get("review_verdict")

    verdict_str = ""
    if verdict and not no_review:
        v = verdict.verdict if hasattr(verdict, 'verdict') else \
            (verdict.get("verdict", "?") if isinstance(verdict, dict) else "?")
        b = len(verdict.blocking) if hasattr(verdict, 'blocking') else 0
        r = len(verdict.required) if hasattr(verdict, 'required') else 0
        s = len(verdict.suggested) if hasattr(verdict, 'suggested') else 0
        verdict_str = f"\n\n## Reviewer Gate\n- **Verdict:** {v}\n- **Blocking:** {b}\n- **Required:** {r}\n- **Suggested:** {s}"

    final_report = draft + verdict_str

    if output:
        with open(output, "w") as f:
            f.write(final_report)
        click.echo(f"Report saved to {output}")
    else:
        click.echo()
        _print_markdown(final_report)


@main.command()
@click.argument("query")
@click.option("--intensity", "-i", type=click.IntRange(1, 5), default=2)
def plan(query, intensity):
    """Generate a research plan without executing research."""
    from ora.graph import build_graph
    graph = build_graph()

    result = _spin(graph.ainvoke(
        {"query": query, "intensity": intensity, "plan_approved": False, "revision_count": 0},
        {"configurable": {"thread_id": "plan-1"}},
    ), message="Generating research plan...")
    plan_text = result.get("research_plan", "No plan generated.")
    click.echo(f"\nResearch Plan for: {query}\n")
    _print_markdown(plan_text)


@main.command()
@click.option("--show", is_flag=True, help="Show current configuration")
@click.option("--init", is_flag=True, help="Create default config at ~/.ora/config.yaml")
def config(show, init):
    """Show or initialize ORA configuration."""
    config_path = os.path.expanduser("~/.ora/config.yaml")

    if init:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        default_config = {
            "models": {
                "default": "deepseek-v4-flash",
                "researcher": "deepseek-v4-flash",
                "supervisor": "deepseek-v4-pro",
                "reviewer": "deepseek-v4-pro",
            },
            "search": {"provider": "firecrawl"},
            "limits": {"max_revisions": 3, "default_intensity": 2},
        }
        with open(config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)
        click.echo(f"Config created at {config_path}")
        return

    settings = load_config()
    click.echo(f"Config file: {config_path}")
    click.echo(f"Default model: {settings.models.default}")
    click.echo(f"Researcher: {settings.models.researcher or '(default)'}")
    click.echo(f"Reviewer: {settings.models.reviewer or '(auto: opposite provider)'}")
    click.echo(f"Search: {settings.search.provider}")
    click.echo(f"Max revisions: {settings.limits.max_revisions}")


@main.command()
def bench():
    """Show instructions for Deep Research Bench submission."""
    click.echo("Deep Research Bench runner (v0)")
    click.echo("  Clone: git clone https://github.com/Ayanami0730/deep_research_bench")
    click.echo("  Then: python tests/run_evaluate.py")
    click.echo("  See docs for full instructions.")


if __name__ == "__main__":
    main()
