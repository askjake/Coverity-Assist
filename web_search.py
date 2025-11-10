# -*- coding: utf-8 -*-
"""
Local web_search helper (incognito-ish, low bandwidth).
- DDG lite query, UA-only, no cookies
- Returns dict with results and (optionally) fetched plaintext per URL
- Meant to be imported by coverity_assist_gateway
"""
import re
from typing import Dict, Any, List
import requests

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118 Safari/537.36"
HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}

def _ddg_lite(query: str, max_results: int = 6, timeout: int = 15) -> List[Dict[str, str]]:
    url = "https://lite.duckduckgo.com/lite/"
    r = requests.get(url, params={"q": query}, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    html_text = r.text
    rows = re.findall(
        r"<tr[^>]*>\\s*<td[^>]*>\\s*<a[^>]+class=['\\\"]result-link['\\\"][^>]+href=['\\\"](.*?)['\\\"][^>]*>(.*?)</a>.*?<td[^>]*>(.*?)</td>",
        html_text, flags=re.S | re.I
    )
    import html as _html
    out = []
    for href, title_html, snippet_html in rows[:max_results]:
        title = _html.unescape(re.sub("<.*?>", "", title_html)).strip() or "(no title)"
        snippet = _html.unescape(re.sub("<.*?>", "", snippet_html)).strip()
        out.append({"title": title, "url": href, "snippet": snippet})
    return out

def do_web_search(query: str, max_results: int = 6, fetch_pages: bool = False) -> Dict[str, Any]:
    results = _ddg_lite(query, max_results=max_results)
    return {
        "query": query,
        "results": results,
        "note": "local web_search helper"
    }
