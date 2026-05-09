"""Firecrawl scrape tool for LangChain."""
from langchain_core.tools import tool


@tool
def scrape_page(url: str) -> str:
    """Scrape a web page for full content using Firecrawl.

    Args:
        url: The URL to scrape.

    Returns:
        Page content as markdown text.
    """
    import os
    from firecrawl import FirecrawlApp

    key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not key:
        return "Error: No Firecrawl API key configured."

    app = FirecrawlApp(api_key=key)
    try:
        result = app.scrape_url(url, params={"formats": ["markdown"]})
        content = result.get("markdown", "")
        if not content:
            return f"No content extracted from {url}"
        if len(content) > 8000:
            content = content[:8000] + "\n\n[Content truncated at 8000 characters]"
        return content
    except Exception as e:
        return f"Scrape error for {url}: {str(e)}"
