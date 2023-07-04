import httpx

def wikipedia(q: str) -> str:
    """
        get the search result of first paragraph of the wikipedia article for the given query.
    """

    return httpx.get("https://en.wikipedia.org/w/api.php", params={
        "action": "query",
        "list": "search",
        "srsearch": q,
        "format": "json"
    }).json()["query"]["search"][0]["snippet"]