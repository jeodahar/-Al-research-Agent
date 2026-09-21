"""Free web search tool (DuckDuckGo) that the CrewAI agent can call."""

from crewai.tools import tool
from ddgs import DDGS

MAX_RESULTS = 5          # keep small so we don't waste Groq tokens
MAX_SNIPPET_CHARS = 300  # trim long snippets for the same reason


@tool("DuckDuckGo Search")
def web_search(query: str) -> str:
    """Search the web with DuckDuckGo.
    Input must be a short search query (plain text).
    Returns a numbered list of results with title, URL and a short snippet."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_RESULTS))
    except Exception as e:
        return f"Search failed: {e}. Try a shorter or different query."

    if not results:
        return "No results found. Try a different query."

    lines = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "")
        url = r.get("href", "")
        snippet = (r.get("body") or "")[:MAX_SNIPPET_CHARS]
        lines.append(f"{i}. {title}\n   URL: {url}\n   {snippet}")
    return "\n\n".join(lines)
