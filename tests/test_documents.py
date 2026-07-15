"""Tests for document segment adapters."""

import pytest
import warnings
from opf._core.documents import (
    DocumentSegment,
    segments_from_strings,
    segments_from_transcript,
    segments_from_subtitles,
    segments_from_ocr,
    segments_from_paths,
    join_segments,
)

def test_segments_from_strings():
    strings = ["Hello world", "Goodbye world"]
    segments = segments_from_strings(strings)
    assert len(segments) == 2
    assert segments[0].text == "Hello world"
    assert segments[0].source_id == "0"
    assert segments[0].order == 0
    assert segments[1].text == "Goodbye world"
    assert segments[1].source_id == "1"
    assert segments[1].order == 1

def test_segments_from_transcript():
    transcript = [
        {"text": "Hello", "start": 0.0, "end": 1.0, "speaker": "A"},
        {"text": "How are you?", "start": 1.0, "end": 2.5, "speaker": "B"},
    ]
    segments = segments_from_transcript(transcript)
    assert len(segments) == 2
    assert segments[0].text == "Hello"
    assert segments[0].source_id == "0"
    assert segments[0].metadata["speaker"] == "A"
    assert segments[1].text == "How are you?"
    assert segments[1].source_id == "1"
    assert segments[1].metadata["speaker"] == "B"

def test_segments_from_subtitles():
    subs = [
        {"text": "Hi", "cue": "cue1", "start": 0},
        {"text": "There", "cue": "cue2", "start": 5},
    ]
    segments = segments_from_subtitles(subs)
    assert len(segments) == 2
    assert segments[0].text == "Hi"
    assert segments[0].source_id == "cue1"
    assert segments[1].text == "There"
    assert segments[1].source_id == "cue2"

def test_segments_from_ocr():
    ocr = {
        "texts": ["Alice", "Bob"],
        "boxes": [[0, 0, 10, 10], [10, 10, 20, 20]],
    }
    segments = segments_from_ocr(ocr)
    assert len(segments) == 2
    assert segments[0].text == "Alice"
    assert segments[0].source_id == "0"
    assert segments[0].metadata["boxes"] == [0, 0, 10, 10]

def test_segments_from_paths():
    payload = {
        "user": {
            "name": "Charlie",
            "bio": "Developer from London",
        }
    }
    paths = [
        ["user", "name"],
        ["user", "bio"],
    ]
    segments = segments_from_paths(payload, paths)
    assert len(segments) == 2
    assert segments[0].text == "Charlie"
    assert segments[0].source_id == "/user/name"
    assert segments[1].text == "Developer from London"
    assert segments[1].source_id == "/user/bio"

def test_join_segments():
    segments = [
        DocumentSegment("Hello", "1", 0, {}),
        DocumentSegment("World", "2", 1, {}),
    ]
    assert join_segments(segments, separator=" ") == "Hello World"
    assert join_segments(segments, separator="\n") == "Hello\nWorld"
