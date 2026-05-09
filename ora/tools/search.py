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
    from ora.config import get_firecrawl_client

    app = get_firecrawl_client()
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
    """Create a configured web search tool (deprecated - use config instead)."""
    return web_search
