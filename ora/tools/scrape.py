"""Firecrawl scrape tool for LangChain."""
import requests
from langchain_core.tools import tool
from ora.config import get_firecrawl_client


@tool
def scrape_page(url: str) -> str:
    """Scrape a web page for full content using Firecrawl.

    Args:
        url: The URL to scrape.

    Returns:
        Page content as markdown text.
    """
    try:
        app = get_firecrawl_client()
        resp = requests.post(
            f"{app.api_url}/v2/scrape",
            json={"url": url, "formats": ["markdown"]},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        data = resp.json()

        if not data.get("success"):
            return f"Scrape failed for {url}: {data}"

        content = data.get("data", {}).get("markdown", "")
        if not content:
            return f"No content extracted from {url}"
        if len(content) > 8000:
            content = content[:8000] + "\n\n[Content truncated at 8000 characters]"
        return content
    except Exception as e:
        return f"Scrape error for {url}: {str(e)}"
