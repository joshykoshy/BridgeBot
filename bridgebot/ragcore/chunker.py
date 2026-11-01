"""Heading-aware chunking of markdown documents.

Concept: retrieval quality depends on what a "chunk" is. v1 cut text every 200 characters,
which split sentences in half and lost which section a passage came from. Here we:
  1. split the document at markdown headings, so a chunk never spans two sections;
  2. pack whole paragraphs into chunks of ~400 tokens with ~60 tokens of overlap;
  3. prefix every chunk with its heading path ("Etiquette in Japan > Gift giving"), so the
     chunk is self-describing for both BM25 and the embedding model;
  4. keep the document metadata (url, title, country, ...) on every chunk for filters and
     citations.

Trade-off: tokens are estimated as words x 1.33 (no tokenizer dependency, fast). It is
approximate; 400 estimated tokens stays under bge-small's 512-wordpiece limit for normal
English. Bigger chunks give more context per hit but blur the embedding; smaller chunks
are precise but lose context. ~400 is a common middle ground.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TOKENS_PER_WORD = 1.33
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    return round(len(text.split()) * TOKENS_PER_WORD)


def split_sections(markdown: str) -> list[tuple[list[str], str]]:
    """Return (heading_path, body) for each section of a markdown document."""
    sections: list[tuple[list[str], str]] = []
    path: list[str] = []
    body: list[str] = []
    for line in markdown.splitlines():
        match = HEADING_RE.match(line)
        if match:
            if "".join(body).strip():
                sections.append((list(path), "\n".join(body).strip()))
            level = len(match.group(1))
            path = path[: level - 1] + [match.group(2).strip()]
            body = []
        else:
            body.append(line)
    if "".join(body).strip():
        sections.append((list(path), "\n".join(body).strip()))
    return sections


def _pack(paragraphs: list[str], max_words: int, overlap_words: int) -> list[str]:
    """Greedily pack paragraphs into windows of <= max_words. Each new window starts with the
    last `overlap_words` of the previous one. A window is only emitted if it contains fresh
    words, so no chunk is ever just a copy of the previous chunk's tail."""
    windows: list[list[str]] = []
    current: list[str] = []
    fresh = 0  # words in `current` that are not carried-over overlap

    def flush() -> None:
        nonlocal current, fresh
        if fresh:
            windows.append(current)
            current = current[-overlap_words:] if overlap_words else []
            fresh = 0

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            continue
        if len(current) + len(words) > max_words:
            flush()
        while len(current) + len(words) > max_words:  # paragraph longer than a window
            room = max_words - len(current)
            current, words = current + words[:room], words[room:]
            fresh += room
            flush()
        current += words
        fresh += len(words)
    flush()
    return [" ".join(w) for w in windows]


def chunk_markdown(
    doc_id: str,
    markdown: str,
    metadata: dict | None = None,
    max_tokens: int = 400,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    """Chunk one markdown document. Chunk ids are stable: '<doc_id>#<n>'."""
    metadata = metadata or {}
    max_words = int(max_tokens / TOKENS_PER_WORD)
    overlap_words = int(overlap_tokens / TOKENS_PER_WORD)
    title = metadata.get("title", doc_id)
    chunks: list[Chunk] = []
    for path, body in split_sections(markdown):
        heading = " > ".join([title] + [h for h in path if h != title])
        paragraphs = re.split(r"\n\s*\n", body)
        for piece in _pack(paragraphs, max_words, overlap_words):
            n = len(chunks)
            chunks.append(Chunk(
                id=f"{doc_id}#{n}",
                text=f"{heading}\n\n{piece}",
                metadata={**metadata, "doc_id": doc_id, "section": heading, "chunk_index": n},
            ))
    return chunks
