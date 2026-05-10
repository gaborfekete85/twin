"""
RAG (Retrieval-Augmented Generation) module.

Crawling strategy (most → least reliable):
  1. Direct HTTP fetch  – works for any public URL (used first)
  2. Tavily extract     – tried when direct fetch fails
  3. Tavily search      – discovers additional pages on the same domain

Flow:
  crawl_website() → chunk_text() → embed_in_batches()
  → save_rag_index() / load_rag_index() → retrieve_context()
"""

import os
import re
import json
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import urlparse, urljoin
from typing import Optional

import numpy as np
import requests

RAG_DIR        = os.getenv("RAG_DIR", "../rag_store")
S3_RAG_PREFIX  = "rag/"

CHUNK_SIZE     = 800
CHUNK_OVERLAP  = 120
MAX_PAGES      = 6
EMBED_BATCH    = 20
EMBED_MODEL    = "text-embedding-3-small"

HTTP_HEADERS   = {"User-Agent": "Mozilla/5.0 (compatible; DigitalTwinBot/1.0)"}
HTTP_TIMEOUT   = 15


# ──────────────────────────────────────────────────────────────────────────────
# HTML → plain text
# ──────────────────────────────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Strip HTML tags and return visible text."""
    SKIP_TAGS = {"script", "style", "nav", "footer", "head", "noscript", "svg", "iframe"}

    def __init__(self):
        super().__init__()
        self.parts:  list[str] = []
        self._depth: int = 0          # nesting depth inside a skip-tag

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._depth:
            self._depth -= 1

    def handle_data(self, data):
        if not self._depth:
            t = data.strip()
            if t:
                self.parts.append(t)


def _html_to_text(html: str) -> str:
    p = _TextExtractor()
    p.feed(html)
    return re.sub(r"\s+", " ", " ".join(p.parts)).strip()


def _meta_refresh_url(html: str, base_url: str) -> Optional[str]:
    """Return the redirect target of a <meta http-equiv='refresh'> tag, or None."""
    m = re.search(r'<meta[^>]+http-equiv=["\']refresh["\'][^>]+content=["\'][^;]*;\s*url=([^"\'>\s]+)',
                  html, re.I)
    if not m:
        m = re.search(r'<meta[^>]+content=["\'][^;]*;\s*url=([^"\'>\s]+)[^>]+http-equiv=["\']refresh["\']',
                      html, re.I)
    if m:
        return urljoin(base_url, m.group(1).strip())
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Direct HTTP fetch (primary strategy)
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_url(url: str, follow_meta_refresh: bool = True) -> Optional[str]:
    """
    Fetch a URL and return its plain-text content, or None on failure.
    Follows meta-refresh redirects one level deep.
    """
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT,
                            allow_redirects=True)
        resp.raise_for_status()

        # Handle meta-refresh redirect (e.g. feketegabor.com → /portfolio)
        if follow_meta_refresh:
            redirect = _meta_refresh_url(resp.text, url)
            if redirect and redirect != url:
                return _fetch_url(redirect, follow_meta_refresh=False)

        text = _html_to_text(resp.text)
        return text if len(text) > 50 else None
    except Exception as exc:
        print(f"[RAG] direct fetch failed for {url}: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Crawling
# ──────────────────────────────────────────────────────────────────────────────

def crawl_website(url: str, tavily_client, max_pages: int = MAX_PAGES) -> list[dict]:
    """
    Crawl a website and return a list of {"url": str, "content": str}.

    Order of attempts:
      1. Direct HTTP fetch of the provided URL (always tried first)
      2. Tavily extract of the same URL (fallback if direct fetch fails)
      3. Tavily search for more pages on the same domain
    """
    results: list[dict] = []
    seen:    set[str]   = set()

    # ── 1. Direct HTTP fetch of the root URL ──────────────────────────────────
    text = _fetch_url(url)
    if text:
        results.append({"url": url, "content": text})
        seen.add(url)
    else:
        # ── 2. Tavily extract fallback ─────────────────────────────────────
        try:
            extracted = tavily_client.extract(urls=[url])
            for r in extracted.get("results", []):
                content  = (r.get("raw_content") or "").strip()
                page_url = r.get("url", url)
                if content and page_url not in seen:
                    results.append({"url": page_url, "content": content})
                    seen.add(page_url)
        except Exception as exc:
            print(f"[RAG] Tavily extract failed for {url}: {exc}")

    # ── 3. Tavily search for more pages on the same domain ───────────────────
    if len(results) < max_pages:
        domain = urlparse(url).netloc
        try:
            searched = tavily_client.search(
                query=f"{domain} site:{domain}",
                max_results=max_pages,
                include_raw_content=True,
                search_depth="advanced",
            )
            for r in searched.get("results", []):
                if len(results) >= max_pages:
                    break
                page_url = r.get("url", "")
                # Try direct fetch first, then fall back to Tavily content
                content  = _fetch_url(page_url) or (r.get("raw_content") or r.get("content", "")).strip()
                if content and page_url not in seen:
                    results.append({"url": page_url, "content": content})
                    seen.add(page_url)
        except Exception as exc:
            print(f"[RAG] Tavily search failed for {domain}: {exc}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks at sentence boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            bp = text.rfind(".", start + overlap, end)
            if bp == -1:
                bp = text.rfind(" ", start + overlap, end)
            if bp > start:
                end = bp + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], openai_client) -> list[list[float]]:
    if not texts:
        return []
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def embed_in_batches(texts: list[str], openai_client, batch_size: int = EMBED_BATCH) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        embeddings.extend(embed_texts(texts[i : i + batch_size], openai_client))
    return embeddings


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def _index_key(training_id: str) -> str:
    return f"{S3_RAG_PREFIX}{training_id}.json"


def save_rag_index(training_id, url, chunks, use_s3, s3_client=None, s3_bucket=""):
    payload = json.dumps({
        "training_id": training_id,
        "url": url,
        "created_at": datetime.now().isoformat(),
        "chunks": chunks,
    })
    if use_s3 and s3_client:
        s3_client.put_object(
            Bucket=s3_bucket, Key=_index_key(training_id),
            Body=payload, ContentType="application/json",
        )
    else:
        os.makedirs(RAG_DIR, exist_ok=True)
        with open(os.path.join(RAG_DIR, f"{training_id}.json"), "w") as fh:
            fh.write(payload)


def load_rag_index(training_id, use_s3, s3_client=None, s3_bucket="") -> Optional[dict]:
    if use_s3 and s3_client:
        try:
            obj = s3_client.get_object(Bucket=s3_bucket, Key=_index_key(training_id))
            return json.loads(obj["Body"].read().decode("utf-8"))
        except Exception:
            return None
    else:
        path = os.path.join(RAG_DIR, f"{training_id}.json")
        return json.load(open(path)) if os.path.exists(path) else None


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────────────────

def _cosine(a, b) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))


def retrieve_context(query, training_id, openai_client, use_s3,
                     s3_client=None, s3_bucket="", top_k=3) -> list[str]:
    index = load_rag_index(training_id, use_s3, s3_client, s3_bucket)
    if not index or not index.get("chunks"):
        return []

    query_emb = embed_texts([query], openai_client)[0]
    scored = [
        (_cosine(query_emb, c["embedding"]), c["text"], c.get("source_url", ""))
        for c in index["chunks"]
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [f"[Source: {src}]\n{text}" for _, text, src in scored[:top_k]]
