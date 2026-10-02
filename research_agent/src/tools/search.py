from langchain_core.tools import tool
from duckduckgo_search import DDGS
import logging

logger = logging.getLogger(__name__)


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo for up-to-date information.

    Args:
        query: The search query string.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return f"No search results found for query: {query}"
            
            formatted_results = []
            for r in results:
                title = r.get("title", "No Title")
                href = r.get("href", "No URL")
                body = r.get("body", "No Description")
                formatted_results.append(f"Title: {title}\nURL: {href}\nSnippet: {body}\n")
            
            return "\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error executing web search for '{query}': {e}")
        return f"Web search encountered an error: {str(e)}. Please try a different query or approach."
