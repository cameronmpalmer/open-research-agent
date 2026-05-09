"""Source evaluation using CRAAP dimensions."""
from ora.state import Source


def compute_craap_score(dimensions: dict) -> int:
    """Sum CRAAP dimensions (each 1-5) into a 5-25 score."""
    return sum(dimensions.get(d, 3) for d in ["currency", "relevance", "authority", "accuracy", "purpose"])


def rate_reliability(craap_score: int) -> str:
    """Convert CRAAP total score to reliability rating.

    High: 19-25, Medium: 12-18, Low: 5-11
    """
    if craap_score >= 19:
        return "High"
    elif craap_score >= 12:
        return "Medium"
    else:
        return "Low"


def evaluate_source(
    url: str,
    title: str = "",
    content: str = "",
    source_type: str = "unknown",
    publication_date: str = "",
) -> Source:
    """Evaluate a source using CRAAP criteria heuristics.

    This is a heuristic-based evaluation for v0. Each dimension is
    scored 1-5 based on domain signals, content length, and source type.
    The overall reliability is derived from the total CRAAP score:
    High: 19-25, Medium: 12-18, Low: 5-11.
    """
    dimensions = {
        "currency": _rate_currency(url, publication_date, source_type),
        "relevance": _rate_relevance(content),
        "authority": _rate_authority(url, source_type),
        "accuracy": _rate_accuracy(content),
        "purpose": _rate_purpose(url, content, source_type),
    }

    score = compute_craap_score(dimensions)
    reliability = rate_reliability(score)

    return Source(
        url=url,
        title=title,
        publication_date=publication_date or None,
        source_type=source_type,  # type: ignore
        currency=dimensions["currency"],
        relevance=dimensions["relevance"],
        authority=dimensions["authority"],
        accuracy=dimensions["accuracy"],
        purpose=dimensions["purpose"],
        overall_reliability=reliability,  # type: ignore
        notes=f"CRAAP score: {score}/25",
    )


def _rate_currency(url: str, date: str, source_type: str) -> int:
    if date:
        return 4
    if any(url.startswith(p) for p in ["https://arxiv.org", "https://doi.org"]):
        return 4
    if ".gov" in url or ".edu" in url:
        return 3
    return 2


def _rate_relevance(content: str) -> int:
    if len(content) > 2000:
        return 4
    elif len(content) > 500:
        return 3
    elif len(content) > 100:
        return 2
    return 1


def _rate_authority(url: str, source_type: str) -> int:
    if source_type in ("academic_paper", "official_doc"):
        return 5
    if ".gov" in url:
        return 5
    if ".edu" in url:
        return 4
    if source_type == "news":
        return 3
    if source_type == "blog":
        return 2
    if source_type in ("forum", "social_media"):
        return 1
    return 2


def _rate_accuracy(content: str) -> int:
    if "http" in content and len(content) > 1000:
        return 4
    if len(content) > 500:
        return 3
    return 2


def _rate_purpose(url: str, content: str, source_type: str) -> int:
    if source_type in ("academic_paper", "official_doc"):
        return 5
    if ".gov" in url or ".edu" in url:
        return 4
    return 2
