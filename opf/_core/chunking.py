"""Tokenizer-aware text chunking with overlap for long-text inference.

This module provides a modality-neutral chunking strategy that:
- Respects token budgets using the model's actual tokenizer
- Prefers boundaries at paragraphs, then sentences, then whitespace
- Adds configurable overlap to catch cross-boundary entities
- Projects local chunk spans back to global text offsets
- Deduplicates overlapping predictions deterministically
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Sequence

import tiktoken


@dataclass(frozen=True)
class Chunk:
    """One chunk of text with its global offset mapping."""

    text: str
    global_start: int
    global_end: int
    chunk_index: int


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for the chunker."""

    max_tokens: int
    overlap_tokens: int
    encoding: tiktoken.Encoding
    # Boundary preferences (in order of preference)
    prefer_paragraph_boundaries: bool = True
    prefer_sentence_boundaries: bool = True
    # Fallback: hard cut at token boundary if no natural boundary found
    hard_cut_fallback: bool = True


# Sentence boundary regex - avoids common abbreviations
_SENTENCE_END_RE = re.compile(
    r"""
    (?<![A-Z]\.)      # Not after single capital letter (e.g. "U.S.")
    (?<!\w\.\w\.)     # Not after "e.g." or "i.e."
    (?<![A-Z][a-z]\.) # Not after "Mr." "Dr." "St." etc.
    (?<=\.|\?|\!)     # After sentence end punctuation
    \s+
    """,
    re.VERBOSE,
)

# Paragraph boundary: double newline or more
_PARAGRAPH_BOUNDARY_RE = re.compile(r"\n\s*\n")


def _find_paragraph_boundaries(text: str) -> list[int]:
    """Return character offsets of paragraph boundaries (after the newlines)."""
    boundaries = [0]
    for match in _PARAGRAPH_BOUNDARY_RE.finditer(text):
        boundaries.append(match.end())
    boundaries.append(len(text))
    return boundaries


def _find_sentence_boundaries(text: str) -> list[int]:
    """Return character offsets of sentence boundaries."""
    boundaries = [0]
    for match in _SENTENCE_END_RE.finditer(text):
        boundaries.append(match.end())
    if boundaries[-1] != len(text):
        boundaries.append(len(text))
    return boundaries


def _tokenize(text: str, encoding: tiktoken.Encoding) -> list[int]:
    """Tokenize text and return token IDs."""
    return encoding.encode(text, allowed_special="all")


def _token_count(text: str, encoding: tiktoken.Encoding) -> int:
    """Return token count for text."""
    return len(_tokenize(text, encoding))


def _decode_tokens(token_ids: Sequence[int], encoding: tiktoken.Encoding) -> str:
    """Decode token IDs back to text."""
    return encoding.decode(list(token_ids))


def _find_cut_point(
    text: str,
    target_token_count: int,
    encoding: tiktoken.Encoding,
    config: ChunkingConfig,
) -> int:
    """Find the best character offset to cut at or before target_token_count.

    Prefers (in order): paragraph boundary, sentence boundary, whitespace, hard token cut.
    Returns character offset in text.
    """
    if target_token_count >= _token_count(text, encoding):
        return len(text)

    # Binary search for token boundary near target
    tokens = _tokenize(text, encoding)
    if target_token_count >= len(tokens):
        return len(text)

    # Decode up to target token to get approximate char position
    approx_text = _decode_tokens(tokens[:target_token_count], encoding)
    approx_char = len(approx_text)

    # Search window around approximate position
    search_start = max(0, approx_char - 200)
    search_end = min(len(text), approx_char + 200)
    window_text = text[search_start:search_end]

    # Try paragraph boundaries first
    if config.prefer_paragraph_boundaries:
        for match in _PARAGRAPH_BOUNDARY_RE.finditer(window_text):
            abs_pos = search_start + match.end()
            # Only accept if it's at or before target (with small tolerance)
            if abs_pos <= approx_char + 50:
                return abs_pos

    # Try sentence boundaries
    if config.prefer_sentence_boundaries:
        for match in _SENTENCE_END_RE.finditer(window_text):
            abs_pos = search_start + match.end()
            if abs_pos <= approx_char + 50:
                return abs_pos

    # Try whitespace boundary (scan backward from approx position)
    rel_approx = approx_char - search_start
    for i in range(min(len(window_text) - 1, rel_approx + 50), -1, -1):
        if window_text[i].isspace():
            return search_start + i + 1

    # Hard token cut fallback
    if config.hard_cut_fallback:
        return search_start + min(len(window_text), max(0, rel_approx))

    return len(text)


def chunk_text(text: str, config: ChunkingConfig) -> list[Chunk]:
    """Split text into overlapping chunks respecting token budget and boundaries.

    Args:
        text: Input text to chunk.
        config: Chunking configuration with tokenizer and limits.

    Returns:
        List of Chunk objects with text and global offset mappings.
    """
    if not text:
        return [Chunk(text="", global_start=0, global_end=0, chunk_index=0)]

    total_tokens = _token_count(text, config.encoding)
    if total_tokens <= config.max_tokens:
        return [Chunk(text=text, global_start=0, global_end=len(text), chunk_index=0)]

    chunks: list[Chunk] = []
    chunk_index = 0
    cursor = 0

    while cursor < len(text):
        # Determine chunk end
        remaining_text = text[cursor:]
        remaining_tokens = _token_count(remaining_text, config.encoding)

        if remaining_tokens <= config.max_tokens:
            # Final chunk
            chunk_text = remaining_text
            global_start = cursor
            global_end = len(text)
        else:
            # Find cut point at max_tokens
            cut_char = _find_cut_point(
                remaining_text, config.max_tokens, config.encoding, config
            )
            chunk_text = remaining_text[:cut_char]
            global_start = cursor
            global_end = cursor + cut_char

        chunks.append(
            Chunk(
                text=chunk_text,
                global_start=global_start,
                global_end=global_end,
                chunk_index=chunk_index,
            )
        )

        # Move cursor forward by (chunk_tokens - overlap_tokens)
        chunk_tokens = _token_count(chunk_text, config.encoding)
        if chunk_tokens <= config.overlap_tokens:
            # Chunk too small, advance by at least 1 token worth
            advance_chars = _find_cut_point(
                remaining_text, max(1, config.overlap_tokens + 1), config.encoding, config
            )
            if advance_chars == 0:
                advance_chars = len(remaining_text)
        else:
            # Find position after (chunk_tokens - overlap_tokens) tokens
            advance_tokens = chunk_tokens - config.overlap_tokens
            advance_text = _decode_tokens(
                _tokenize(chunk_text, config.encoding)[:advance_tokens], config.encoding
            )
            advance_chars = len(advance_text)

        # Ensure we make progress
        if advance_chars <= 0:
            advance_chars = 1

        cursor += advance_chars
        chunk_index += 1

    return chunks


def map_local_spans_to_global(
    spans: Sequence[tuple[int, int, int]],  # (label, local_start, local_end)
    chunk: Chunk,
) -> list[tuple[int, int, int]]:
    """Convert chunk-local character spans to global text offsets.

    Args:
        spans: Detected spans in chunk-local coordinates (label, start, end).
        chunk: The chunk these spans came from.

    Returns:
        Spans in global text coordinates (label, global_start, global_end).
    """
    global_spans = []
    for label, local_start, local_end in spans:
        global_start = chunk.global_start + local_start
        global_end = chunk.global_start + local_end
        # Clamp to chunk boundaries (safety)
        if global_start < chunk.global_start:
            global_start = chunk.global_start
        if global_end > chunk.global_end:
            global_end = chunk.global_end
        if global_end > global_start:
            global_spans.append((label, global_start, global_end))
    return global_spans


def merge_overlapping_spans(
    spans: Sequence[tuple[int, int, int]],
    text: str,
    prefer_longer: bool = True,
) -> list[tuple[int, int, int]]:
    """Merge overlapping/duplicate spans from overlapping chunks.

    Args:
        spans: All spans from all chunks in global coordinates (label, start, end).
        text: Original full text (for validation).
        prefer_longer: If overlapping spans have different labels, prefer the longer span.

    Returns:
        Deduplicated, non-overlapping spans sorted by start position.
    """
    if not spans:
        return []

    # Clamp spans to text bounds and filter invalid
    valid_spans = []
    for label, start, end in spans:
        # Clamp to valid range
        clamped_start = max(0, min(start, len(text)))
        clamped_end = max(0, min(end, len(text)))
        if clamped_end > clamped_start:
            valid_spans.append((label, clamped_start, clamped_end))

    # Sort by start, then by negative length (longer first), then by label
    sorted_spans = sorted(
        valid_spans, key=lambda s: (s[1], -(s[2] - s[1]), s[0])
    )

    merged: list[tuple[int, int, int]] = []
    for label, start, end in sorted_spans:
        # Check for overlap with last kept span
        if merged and start < merged[-1][2]:
            last_label, last_start, last_end = merged[-1]
            if end <= last_end:
                # Current span entirely within last span - skip
                continue
            if prefer_longer and (end - start) > (last_end - last_start):
                # Current span is longer - replace last
                merged[-1] = (label, start, end)
            # else: keep last span, skip current (overlaps)
        else:
            merged.append((label, start, end))

    return merged


def build_chunking_config(
    encoding_name: str,
    max_tokens: int = 8192,
    overlap_tokens: int = 128,
    prefer_paragraph_boundaries: bool = True,
    prefer_sentence_boundaries: bool = True,
    hard_cut_fallback: bool = True,
) -> ChunkingConfig:
    """Build a ChunkingConfig with the given parameters."""
    encoding = tiktoken.get_encoding(encoding_name)
    return ChunkingConfig(
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        encoding=encoding,
        prefer_paragraph_boundaries=prefer_paragraph_boundaries,
        prefer_sentence_boundaries=prefer_sentence_boundaries,
        hard_cut_fallback=hard_cut_fallback,
    )