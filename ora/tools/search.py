"""Firecrawl search tool for LangChain."""
import requests
from langchain_core.tools import tool
from ora.config import get_firecrawl_client


@tool
def web_search(query: str) -> str:
    """Search the web using Firecrawl.

    Args:
        query: The search query string.

    Returns:
        Search results as formatted text with URLs and snippets.
    """
    try:
        app = get_firecrawl_client()
        # Use REST API directly since SDK may not expose search
        resp = requests.post(
            f"{app.api_url}/v1/search",
            json={"query": query, "limit": 5},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        data = resp.json()

        if not data.get("success"):
            return f"Search failed: {data}"

        results = data.get("data", [])
        if not results:
            return "No search results found."

        formatted = []
        for i, item in enumerate(results, 1):
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
