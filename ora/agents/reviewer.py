"""Adversarial reviewer agent node for LangGraph."""
import json
from typing import Any
from langchain_core.runnables import RunnableConfig
from ora.state import ResearchState, ReviewVerdict
from ora.prompts import REVIEWER_PROMPT
from ora.config import load_config, get_reviewer_model, get_llm


def parse_reviewer_output(output: str) -> ReviewVerdict:
    """Parse the reviewer's JSON output into a ReviewVerdict.

    Handles JSON in markdown code fences and malformed JSON.
    """
    try:
        # Extract JSON from markdown fences if present
        if "```json" in output:
            start = output.index("```json") + 7
            end = output.index("```", start)
            json_str = output[start:end].strip()
        elif "```" in output:
            start = output.index("```") + 3
            end = output.index("```", start)
            json_str = output[start:end].strip()
        else:
            json_str = output.strip()

        data = json.loads(json_str)
        return ReviewVerdict(
            verdict=data.get("verdict", "PASS"),
            blocking=data.get("blocking", []),
            required=data.get("required", []),
            suggested=data.get("suggested", []),
            contradicting_evidence_found=data.get("contradicting_evidence_found", []),
            confidence_recalibrations=data.get("confidence_recalibrations", {}),
        )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return ReviewVerdict(
            verdict="REVISE",
            blocking=[f"Reviewer output parsing failed: {str(e)}. Raw: {output[:200]}"],
        )


def reviewer_node(
    state: ResearchState, config: RunnableConfig = None
) -> dict[str, Any]:
    """Adversarial reviewer LangGraph node.

    Receives the draft report and original query. Does NOT receive the
    researcher's intermediate findings or search queries. Uses a different
    model provider from the researcher to prevent correlated errors.
    """
    from ora.progress import emit_progress

    settings = load_config()
    model_name = get_reviewer_model(settings)

    emit_progress(config, "Reviewer: auditing draft report...", kind="review")

    llm = get_llm(model_name, temperature=0.2)

    prompt_text = REVIEWER_PROMPT.format(
        query=state.get("query", ""),
        report=state.get("draft_report", ""),
    )

    response = llm.invoke(prompt_text)
    output = response.content if hasattr(response, 'content') else str(response)

    verdict = parse_reviewer_output(output)

    v = verdict.verdict if hasattr(verdict, 'verdict') else "PASS"
    if v == "REVISE":
        emit_progress(
            config,
            "Reviewer: REVISE — restarting research to address gaps",
            kind="warning",
        )
    else:
        emit_progress(config, "Reviewer: PASS — report accepted", kind="success")

    return {
        "review_verdict": verdict,
        "review_verdict_raw": output,
        "revision_count": state.get("revision_count", 0) + 1,
        "messages": [output],
    }
