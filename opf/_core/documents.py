"""Explicit, modality-neutral adapters for redactable document text."""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class DocumentSegment:
    text: str
    source_id: str
    order: int
    metadata: dict[str, Any]
    path: tuple[Any, ...] = ()


def _segment(text: Any, source_id: str, order: int, metadata=None, path=()):
    if not isinstance(text, str):
        raise TypeError("document segment text must be a string")
    return DocumentSegment(text, source_id, order, dict(metadata or {}), tuple(path))


def segments_from_strings(values: Sequence[str]) -> list[DocumentSegment]:
    return [_segment(value, str(i), i) for i, value in enumerate(values)]


def segments_from_transcript(values: Sequence[dict[str, Any]]) -> list[DocumentSegment]:
    result = []
    for i, item in enumerate(values):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            warnings.warn(f"Skipping malformed transcript segment {i}", UserWarning)
            continue
        metadata = {k: copy.deepcopy(v) for k, v in item.items() if k != "text"}
        result.append(_segment(item["text"], str(item.get("id", i)), i, metadata))
    return result


def segments_from_subtitles(values: Sequence[dict[str, Any]]) -> list[DocumentSegment]:
    result = []
    for i, item in enumerate(values):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            warnings.warn(f"Skipping malformed subtitle cue {i}", UserWarning)
            continue
        metadata = {k: copy.deepcopy(v) for k, v in item.items() if k != "text"}
        result.append(_segment(item["text"], str(item.get("id", item.get("cue", i))), i, metadata))
    return result


def segments_from_ocr(payload: dict[str, Any]) -> list[DocumentSegment]:
    """Adapt PaddleOCR-like ``texts`` plus parallel metadata arrays."""
    texts = payload.get("texts", payload.get("text"))
    if not isinstance(texts, list):
        raise ValueError("OCR payload requires a text/texts array")
    arrays = {k: v for k, v in payload.items() if isinstance(v, list) and k not in {"text", "texts"}}
    for key, values in arrays.items():
        if len(values) != len(texts):
            warnings.warn(f"OCR metadata array {key!r} length does not match text array", UserWarning)
    result = []
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            warnings.warn(f"Skipping malformed OCR text at index {i}", UserWarning)
            continue
        metadata = {k: copy.deepcopy(v[i]) for k, v in arrays.items() if i < len(v)}
        result.append(_segment(text, str(i), i, metadata))
    return result


def segments_from_paths(payload: Any, paths: Iterable[Sequence[Any]]) -> list[DocumentSegment]:
    result = []
    for order, path in enumerate(paths):
        current = payload
        try:
            for key in path:
                current = current[key]
        except (KeyError, IndexError, TypeError):
            warnings.warn(f"Text path {tuple(path)!r} was not found", UserWarning)
            continue
        if not isinstance(current, str):
            warnings.warn(f"Text path {tuple(path)!r} is not a string", UserWarning)
            continue
        result.append(_segment(current, "/" + "/".join(map(str, path)), order, path=path))
    return result


def join_segments(segments: Sequence[DocumentSegment], separator: str = "\n") -> str:
    return separator.join(segment.text for segment in segments)