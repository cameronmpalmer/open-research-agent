"""Click CLI for Open Research Agent (ORA)."""
import logging
import os
import re
import warnings
from datetime import datetime

import click
import yaml
from ora.config import get_researcher_model, load_config

logging.captureWarnings(True)
warnings.filterwarnings("ignore", message=".*allowed_objects.*", module="langgraph")


def _slugify(query: str) -> str:
    """Turn a research query into a filesystem-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", query).strip().lower()
    slug = re.sub(r"[-\s]+", "-", slug)
    return slug[:60]


def _auto_filename(query: str) -> str:
    """Generate a timestamped output filename from a query."""
    ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return f"{_slugify(query)}-{ts}.md"


def _spin(func, message="Working..."):
    """Run a sync function with a rich spinner and cycling dots.

    The status message cycles its trailing dots (., .., ...) while
    the function runs in a background thread.
    """
    from rich.console import Console
    import threading
    import time

    base = message.rstrip(".")

    result = None
    exception = None
    done = threading.Event()

    def _worker():
        nonlocal result, exception
        try:
            result = func()
        except Exception as exc:
            exception = exc
        finally:
            done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    dots = 0
    console = Console()
    with console.status(f"{base}.", spinner="dots") as status:
        while not done.wait(timeout=0.4):
            dots = (dots + 1) % 3
            suffix = "." * (dots + 1)
            status.update(f"{base}{suffix}")

    thread.join()

    if exception:
        raise exception
    return result


def _print_markdown(text: str):
    """Render markdown to the terminal using rich."""
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.style import Style
    console = Console(style=Style(bgcolor=None))
    console.print(Markdown(text))


def _format_progress_event(event: dict) -> str:
    """Format a progress event for line-by-line verbose output."""
    icons = {
        "search": "🔎",
        "scrape": "🌐",
        "success": "✓",
        "error": "✗",
        "write": "✍️",
        "info": "•",
    }
    icon = icons.get(event.get("kind"), "•")
    return f"{icon} {event.get('message', '')}"


def _print_progress_event(event: dict) -> None:
    """Print a progress event immediately."""
    click.echo(_format_progress_event(event))


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
              help="Explicit output file path (overrides auto-named save)")
@click.option("--stdout", is_flag=True,
              help="Also print the report to stdout")
@click.option("--no-save", is_flag=True,
              help="Do not save to file; print to stdout only")
@click.option("--model", "-m", default=None,
              help="LLM model for researcher/writer (e.g., openai:gpt-4.1)")
@click.option("--reviewer-model", "-r", default=None,
              help="LLM model for adversarial reviewer")
@click.option("--max-revisions", type=int, default=3,
               help="Max writer-reviewer revision cycles")
@click.option("--no-review", is_flag=True,
               help="Skip adversarial review")
@click.option("--quiet", is_flag=True,
              help="Disable live progress indicators and use spinner-only output")
@click.option("--verbose", is_flag=True, hidden=True)
@click.option("-y", "--auto-approve", is_flag=True,
              help="Auto-approve the research plan and execute immediately")
@click.option("--hide-plan-on-autoapprove", is_flag=True,
              help="Suppress plan output when --auto-approve is used")
def research(query, intensity, output, model, reviewer_model, max_revisions,
             no_review, quiet, verbose, stdout, no_save, auto_approve,
             hide_plan_on_autoapprove):
    """Run the full research pipeline."""
    if intensity > 3:
        click.echo("Levels 4-5 not yet implemented. Falling back to Level 3.", err=True)
        intensity = 3

    click.echo(f"ORA Research: {query}")
    settings = load_config()
    researcher_model_name = model or settings.models.researcher or settings.models.default
    reviewer_model_name = reviewer_model or settings.models.reviewer or "deepseek-v4-pro"
    click.echo(f"  Intensity: {intensity} | Researcher: {researcher_model_name} | Reviewer: {reviewer_model_name}")

    if model:
        settings.models.researcher = model
    if reviewer_model:
        settings.models.reviewer = reviewer_model

    from ora.graph import build_plan_graph, build_research_graph

    # Phase 1: Generate and review research plan
    plan_graph = build_plan_graph()
    initial_state = {
        "query": query,
        "intensity": intensity,
        "plan_approved": False,
        "revision_count": 0,
    }

    plan_result = _spin(lambda: plan_graph.invoke(initial_state), message="Generating research plan...")
    plan = plan_result.get("research_plan", "No plan generated.")

    if not (auto_approve and hide_plan_on_autoapprove):
        _print_markdown(plan)

    if auto_approve:
        click.echo()
    else:
        from ora.agents.supervisor import revise_plan_text

        while True:
            choice = click.prompt(
                "\n  [A]pprove and run  [E]dit  [R]evise  [C]ancel",
                type=click.Choice(["A", "E", "R", "C"], case_sensitive=False),
                default="A",
                show_choices=False,
                show_default=False,
            ).upper()

            if choice == "A":
                break
            elif choice == "C":
                click.echo("  Research cancelled.")
                return
            elif choice == "E":
                edited = click.edit(text=plan, extension=".md")
                if edited is None:
                    click.echo("  Edit cancelled, plan unchanged.")
                    continue
                plan = plan_result["research_plan"] = edited.rstrip("\n") + "\n"
                _print_markdown(plan)
            elif choice == "R":
                feedback = click.prompt("  Feedback for supervisor")
                plan = plan_result["research_plan"] = _spin(
                    lambda: revise_plan_text(query, intensity, plan, feedback),
                    message="Revising plan...",
                )
                _print_markdown(plan)

    # Phase 2: Run research pipeline
    click.echo()
    research_graph = build_research_graph()
    plan_result["plan_approved"] = True
    if quiet:
        final_state = _spin(lambda: research_graph.invoke(plan_result), message="Researching...")
    else:
        final_state = research_graph.invoke(
            plan_result,
            {"configurable": {"progress_callback": _print_progress_event}},
        )

    if no_review:
        click.echo("  Note: --no-review is currently ignored because the reviewer is disabled.", err=True)
    if max_revisions != 3:
        click.echo("  Note: --max-revisions is currently ignored because the reviewer is disabled.", err=True)

    sources_count = len(final_state.get("sources") or [])
    findings_count = len(final_state.get("findings") or [])
    draft_len = len(final_state.get("draft_report") or "")
    click.echo(f"  Sources: {sources_count} | Findings: {findings_count} | Draft: {draft_len} chars")

    if not final_state.get("draft_report"):
        click.echo("  ⚠️  No report was generated. Check your Firecrawl and API key configuration.", err=True)
        return

    # Reviewer is disabled in the current graph, so use the draft as-is.
    draft = final_state.get("draft_report", "No report generated.")

    # Determine output path
    save_path = output or (None if no_save else _auto_filename(query))

    # Save to file if a path was determined
    if save_path:
        with open(save_path, "w") as f:
            f.write(draft)
        click.echo(f"Report saved to {save_path}")

    # Print to stdout if requested (or if no-save with no output path)
    if stdout or no_save:
        click.echo()
        _print_markdown(draft)


@main.command()
@click.argument("query")
@click.option("--intensity", "-i", type=click.IntRange(1, 5), default=2)
def plan(query, intensity):
    """Generate a research plan without executing research."""
    if intensity > 3:
        click.echo("Levels 4-5 not yet implemented. Falling back to Level 3.", err=True)
        intensity = 3

    from ora.graph import build_plan_graph
    graph = build_plan_graph()

    result = _spin(lambda: graph.invoke(
        {"query": query, "intensity": intensity, "plan_approved": False, "revision_count": 0},
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
        if os.path.exists(config_path):
            if not click.confirm(f"Config already exists at {config_path}. Overwrite?", prompt_suffix=" [y/n]: "):
                click.echo("Aborted.")
                return
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        default_config = {
            "models": {
                "default": "deepseek-v4-flash",
                "researcher": "deepseek-v4-flash",
                "supervisor": "deepseek-v4-pro",
                "reviewer": "deepseek-v4-pro",
            },
            "search": {
                "provider": "firecrawl",
                "firecrawl_api_url": "https://api.firecrawl.com",
            },
            "limits": {"max_revisions": 3, "default_intensity": 2},
        }
        with open(config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)
        click.echo(f"Config created at {config_path}")
        return

    settings = load_config()
    click.echo(f"Config file: {config_path}")
    click.echo()
    click.echo(f"Supervisor (planning & routing): {settings.models.supervisor or 'deepseek-v4-pro'}")
    click.echo(f"Researcher (web search & source eval): {get_researcher_model(settings)}")
    click.echo(f"Writer (report synthesis): {settings.models.researcher or settings.models.default}")
    click.echo(f"Reviewer (adversarial audit): {settings.models.reviewer or 'deepseek-v4-pro'}")
    click.echo()
    click.echo(f"Search backend: {settings.search.provider}")
    click.echo(f"Firecrawl URL: {settings.search.firecrawl_api_url}")
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
