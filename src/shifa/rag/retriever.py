"""Clinic KB retrieval: markdown split on '## ' headings, Gemini embeddings (free tier), Chroma.

Each '## ' section is one chunk with a stable id ('<file>#<heading>') used for citations.
Sections are small and self-contained on purpose, so a chunk never mixes two doctors' fees.

Why an API embedder instead of a local model: on a 120 KB/s connection, torch + a local model is
~700 MB of downloads. Documents are embedded once at ingest and cached in Chroma, so only the
query embedding is a network call at runtime.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb

from shifa.config import get_settings

COLLECTION = "clinic_kb"


@dataclass
class Chunk:
    id: str
    text: str
    score: float = 0.0


def split_markdown(path: Path) -> list[Chunk]:
    chunks = []
    for section in re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.M)[1:]:
        heading, _, body = section.partition("\n")
        heading = heading.strip()
        slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
        chunks.append(Chunk(id=f"{path.stem}#{slug}", text=f"{heading}: {body.strip()}"))
    return chunks


@lru_cache
def _client():
    from google import genai

    return genai.Client(api_key=get_settings().google_api_key)


def embed(texts: list[str], kind: str) -> list[list[float]]:
    """kind: 'query' or 'passage'. gemini-embedding-2 takes the task as a text prefix."""
    from google.genai import types

    prefixed = [f"task: search result | query: {t}" if kind == "query" else f"title: none | text: {t}"
                for t in texts]
    # One Content per text; passing a plain list would return a single aggregated embedding.
    res = _client().models.embed_content(
        model=get_settings().embedding_model,
        contents=[types.Content(parts=[types.Part.from_text(text=t)]) for t in prefixed],
        config=types.EmbedContentConfig(output_dimensionality=768))
    return [e.values for e in res.embeddings]


@lru_cache
def _collection():
    client = chromadb.PersistentClient(path=str(get_settings().chroma_path))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def ingest(kb_dir: Path | None = None) -> int:
    kb_dir = kb_dir or get_settings().kb_dir
    chunks = [c for p in sorted(kb_dir.glob("*.md")) for c in split_markdown(p)]
    col = _collection()
    existing = col.get()["ids"]
    if existing:
        col.delete(ids=existing)
    col.add(ids=[c.id for c in chunks], documents=[c.text for c in chunks],
            embeddings=embed([c.text for c in chunks], "passage"))
    return len(chunks)


def search(query: str, k: int = 4, min_score: float = 0.0) -> list[Chunk]:
    """Cosine similarity. min_score starts permissive; tune it once we see real score distributions."""
    col = _collection()
    if col.count() == 0:
        ingest()
    res = col.query(query_embeddings=embed([query], "query"), n_results=k)
    hits = [Chunk(id=i, text=d, score=1 - dist)
            for i, d, dist in zip(res["ids"][0], res["documents"][0], res["distances"][0])]
    return [h for h in hits if h.score >= min_score]


if __name__ == "__main__":
    print(f"Ingested {ingest()} chunks")
