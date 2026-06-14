"""
Web search abstraction layer.

Supported backends (configured via config.toml):
  web_search_provider = "duckduckgo"  # default, no API key required
  web_search_provider = "brave"       # requires brave_search_api_key
  web_search_provider = "serpapi"     # requires serpapi_api_key

Each backend returns a list of SearchResult dicts:
  {"title": str, "url": str, "snippet": str}
"""

import re
import time
from typing import Optional

import requests
from loguru import logger

from app.config import config

_DEFAULT_TIMEOUT = 10
_MAX_RESULTS = 5


class SearchResult:
    def __init__(self, title: str, url: str, snippet: str):
        self.title = title
        self.url = url
        self.snippet = snippet

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}

    def to_text(self) -> str:
        return f"[{self.title}]\n{self.snippet}\nURL: {self.url}"


def _search_duckduckgo(query: str, max_results: int = _MAX_RESULTS) -> list[SearchResult]:
    """DuckDuckGo HTML search — no API key required."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    }
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query, "kl": "jp-ja"},
            headers=headers,
            timeout=_DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []

    results = []
    # Extract result blocks: <div class="result__body"> ... </div>
    body_blocks = re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</span>',
        resp.text,
        re.DOTALL,
    )
    for url, title_html, snippet_html in body_blocks[:max_results]:
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        snippet = re.sub(r"<[^>]+>", "", snippet_html).strip()
        # DDG redirects — extract actual URL
        real_url = re.sub(r".*uddg=([^&]+).*", lambda m: requests.utils.unquote(m.group(1)), url)
        if title and snippet:
            results.append(SearchResult(title=title, url=real_url, snippet=snippet))

    logger.debug(f"DuckDuckGo search '{query}' → {len(results)} results")
    return results


def _search_brave(query: str, max_results: int = _MAX_RESULTS) -> list[SearchResult]:
    """Brave Search API — requires brave_search_api_key in config."""
    api_key = config.app.get("brave_search_api_key", "")
    if not api_key:
        logger.warning("brave_search_api_key not set, falling back to DuckDuckGo")
        return _search_duckduckgo(query, max_results)

    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results, "search_lang": "ja"},
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            timeout=_DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Brave search failed: {e}")
        return _search_duckduckgo(query, max_results)

    results = []
    for item in data.get("web", {}).get("results", [])[:max_results]:
        results.append(SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("description", ""),
        ))
    return results


def _search_serpapi(query: str, max_results: int = _MAX_RESULTS) -> list[SearchResult]:
    """SerpAPI Google Search — requires serpapi_api_key in config."""
    api_key = config.app.get("serpapi_api_key", "")
    if not api_key:
        logger.warning("serpapi_api_key not set, falling back to DuckDuckGo")
        return _search_duckduckgo(query, max_results)

    try:
        resp = requests.get(
            "https://serpapi.com/search",
            params={"q": query, "api_key": api_key, "num": max_results, "hl": "ja"},
            timeout=_DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"SerpAPI search failed: {e}")
        return _search_duckduckgo(query, max_results)

    results = []
    for item in data.get("organic_results", [])[:max_results]:
        results.append(SearchResult(
            title=item.get("title", ""),
            url=item.get("link", ""),
            snippet=item.get("snippet", ""),
        ))
    return results


def search(query: str, max_results: int = _MAX_RESULTS) -> list[SearchResult]:
    """
    Execute a web search using the configured provider.
    Falls back to DuckDuckGo if provider is not recognized.
    """
    provider = config.app.get("web_search_provider", "duckduckgo").lower()
    logger.info(f"[web_search] provider={provider} query='{query}'")

    if provider == "brave":
        results = _search_brave(query, max_results)
    elif provider == "serpapi":
        results = _search_serpapi(query, max_results)
    else:
        results = _search_duckduckgo(query, max_results)

    # Rate-limit courtesy delay
    time.sleep(0.5)
    return results


def format_results_as_context(results: list[SearchResult], query: str) -> str:
    """Format search results into a readable context block for LLM."""
    if not results:
        return f"[Search: '{query}' — no results found]"
    lines = [f"[Search results for: '{query}']"]
    for i, r in enumerate(results, 1):
        lines.append(f"\n--- Result {i} ---")
        lines.append(r.to_text())
    return "\n".join(lines)
