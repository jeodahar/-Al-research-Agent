"""Free web tools (DuckDuckGo search + page reader) that the CrewAI agent can call."""

import requests
from bs4 import BeautifulSoup
from crewai.tools import tool
from ddgs import DDGS

MAX_RESULTS = 5          # keep small so we don't waste Groq tokens
MAX_SNIPPET_CHARS = 300  # trim long snippets for the same reason
MAX_PAGE_CHARS = 3000    # how much text we return from one web page


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


@tool("Read Webpage")
def read_webpage(url: str) -> str:
    """Open a web page and return its main text (shortened).
    Input must be a full URL starting with http:// or https:// taken from the
    search results. PDF files are not supported."""
    if not url.startswith(("http://", "https://")):
        return "Invalid URL. It must start with http:// or https://"
    if url.lower().split("?")[0].endswith(".pdf"):
        return "PDF files can't be read. Use the search snippets or try a different page."

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (research-agent)"},
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as e:
        return f"Could not open the page: {e}"

    if "pdf" in resp.headers.get("Content-Type", "").lower():
        return "PDF files can't be read. Use the search snippets or try a different page."

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())

    if not text:
        return "The page had no readable text."
    return text[:MAX_PAGE_CHARS]
