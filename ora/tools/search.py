"""Firecrawl search tool for LangChain."""
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web using Firecrawl.

    Args:
        query: The search query string.

    Returns:
        Search results as formatted text with URLs and snippets.
    """
    import os
    from firecrawl import FirecrawlApp

    key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not key:
        return "Error: No Firecrawl API key configured. Set FIRECRAWL_API_KEY."

    app = FirecrawlApp(api_key=key)
    try:
        results = app.search(query, params={"limit": 5})
        if not results or "data" not in results:
            return "No search results found."

        formatted = []
        for i, item in enumerate(results.get("data", []), 1):
            url = item.get("url", "")
            title = item.get("title", "Untitled")
            snippet = item.get("description", "")[:300]
            formatted.append(f"{i}. [{title}]({url})\n   {snippet}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {str(e)}"


def create_search_tool(api_key: str = ""):
    """Create a configured web search tool.

    Args:
        api_key: Firecrawl API key. If empty, reads FIRECRAWL_API_KEY env var.

    Returns:
        Configured LangChain tool.

    Raises:
        ValueError: If no API key is provided or found in env.
    """
    import os
    if not api_key:
        api_key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Firecrawl api_key is required. Set FIRECRAWL_API_KEY or pass api_key."
        )
    os.environ["FIRECRAWL_API_KEY"] = api_key
    return web_search
