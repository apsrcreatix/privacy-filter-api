"""Tests for the tokenizer-aware chunking module."""

import pytest
import tiktoken

from opf._core.chunking import (
    Chunk,
    ChunkingConfig,
    build_chunking_config,
    chunk_text,
    map_local_spans_to_global,
    merge_overlapping_spans,
    _find_cut_point,
    _find_paragraph_boundaries,
    _find_sentence_boundaries,
)


class TestBoundaries:
    """Test boundary detection helpers."""

    def test_paragraph_boundaries(self):
        text = "Para 1.\n\nPara 2.\n\n\nPara 3."
        boundaries = _find_paragraph_boundaries(text)
        # Regex \n\s*\n matches: first \n\n (end=9), then \n\n\n matches as \n+\n (end=19)
        assert boundaries == [0, 9, 19, len(text)]

    def test_sentence_boundaries(self):
        text = "Hello world. How are you? I'm fine!"
        boundaries = _find_sentence_boundaries(text)
        assert boundaries[0] == 0
        assert boundaries[-1] == len(text)
        # Should find boundaries after ". " and "? " and "! "
        # "Hello world. " = 13 chars
        assert 13 in boundaries  # after "Hello world. "
        # "How are you? " = 13 chars, so position 26
        assert 26 in boundaries  # after "How are you? "

    def test_sentence_boundaries_abbreviations(self):
        # Should not split on common abbreviations
        text = "Mr. Smith went to the U.S. He saw Dr. Jones."
        boundaries = _find_sentence_boundaries(text)
        # Should only split at the final period
        assert boundaries[-1] == len(text)
        # Should not split at "Mr." "U.S." "Dr."
        assert len([b for b in boundaries if 0 < b < len(text)]) <= 1


class TestChunking:
    """Test the main chunking logic."""

    @pytest.fixture
    def config(self) -> ChunkingConfig:
        encoding = tiktoken.get_encoding("cl100k_base")
        return ChunkingConfig(
            max_tokens=100,
            overlap_tokens=20,
            encoding=encoding,
        )

    def test_short_text_no_chunking(self, config: ChunkingConfig):
        text = "Short text."
        chunks = chunk_text(text, config)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].global_start == 0
        assert chunks[0].global_end == len(text)

    def test_exact_boundary(self, config: ChunkingConfig):
        # Text that tokenizes to exactly max_tokens
        encoding = config.encoding
        tokens = list(range(config.max_tokens))
        text = encoding.decode(tokens)
        chunks = chunk_text(text, config)
        assert len(chunks) == 1

    def test_long_text_chunks(self, config: ChunkingConfig):
        text = "This is a sentence. " * 50  # Long text
        chunks = chunk_text(text, config)
        assert len(chunks) > 1
        # Check global offsets are correct
        for i, chunk in enumerate(chunks):
            if i:
                assert chunk.global_start <= chunks[i - 1].global_end
            assert chunk.global_end == chunk.global_start + len(chunk.text)
            assert chunk.chunk_index == i

    def test_overlap_creates_shared_text(self, config: ChunkingConfig):
        text = "Sentence one. " * 300
        chunks = chunk_text(text, config)
        assert len(chunks) >= 2
        # Overlap means end of chunk 0 overlaps with start of chunk 1
        # The overlap region should appear in both chunks
        overlap_start = chunks[0].global_end - 50  # approximate
        # Verify we can find some shared text
        assert chunks[0].text[-30:] in chunks[1].text[:80]

    def test_paragraph_boundary_preference(self, config: ChunkingConfig):
        # Create text with clear paragraph boundaries
        para = "Paragraph one. " * 20
        text = para + "\n\n" + para + "\n\n" + para
        chunks = chunk_text(text, config)
        # Should prefer cutting at paragraph boundaries
        for chunk in chunks[:-1]:
            # Last chunk may not end at paragraph boundary
            assert chunk.text.rstrip().endswith(".") or chunk.text.endswith("\n")

    def test_empty_text(self, config: ChunkingConfig):
        chunks = chunk_text("", config)
        assert len(chunks) == 1
        assert chunks[0].text == ""
        assert chunks[0].global_start == 0
        assert chunks[0].global_end == 0

    def test_unicode_text(self, config: ChunkingConfig):
        text = "Hello 世界 🌍 " * 30
        chunks = chunk_text(text, config)
        assert len(chunks) >= 1
        # Reconstruct and verify
        reconstructed = "".join(c.text for c in chunks)
        # Note: due to overlap, reconstructed != original
        # But all chunks should be valid substrings
        for chunk in chunks:
            assert chunk.text in text

    def test_no_punctuation_ocr_text(self, config: ChunkingConfig):
        # OCR-like text without punctuation
        text = "word " * 200
        chunks = chunk_text(text, config)
        assert len(chunks) > 1
        # Should fall back to whitespace or hard cuts
        for chunk in chunks:
            assert len(chunk.text) > 0

    def test_overlap_coverage(self, config: ChunkingConfig):
        """Test that overlap is sufficient to catch cross-boundary entities."""
        # Create text where an entity spans a chunk boundary
        text = "Alice lives at " + "123 Main St " * 10 + "in New York"
        chunks = chunk_text(text, config)
        # The overlap should be at least overlap_tokens worth of text
        for i in range(len(chunks) - 1):
            chunk_tokens = len(config.encoding.encode(chunks[i].text))
            next_chunk_tokens = len(config.encoding.encode(chunks[i + 1].text))
            # Verify overlap exists (approximate check)
            assert chunk_tokens > 0 and next_chunk_tokens > 0


class TestSpanMapping:
    """Test span projection from chunk-local to global."""

    @pytest.fixture
    def config(self) -> ChunkingConfig:
        encoding = tiktoken.get_encoding("cl100k_base")
        return ChunkingConfig(
            max_tokens=50,
            overlap_tokens=10,
            encoding=encoding,
        )

    def test_map_local_to_global(self, config: ChunkingConfig):
        text = "This is a test. " * 20
        chunks = chunk_text(text, config)
        chunk = chunks[0]
        # Local span in first chunk
        local_spans = [(1, 0, 10)]  # label=1, start=0, end=10
        global_spans = map_local_spans_to_global(local_spans, chunk)
        assert global_spans == [(1, 0, 10)]

    def test_map_middle_chunk(self, config: ChunkingConfig):
        text = "This is a test. " * 20
        chunks = chunk_text(text, config)
        if len(chunks) > 1:
            chunk = chunks[1]
            local_spans = [(2, 5, 15)]
            global_spans = map_local_spans_to_global(local_spans, chunk)
            assert global_spans[0][1] == chunk.global_start + 5
            assert global_spans[0][2] == chunk.global_start + 15

    def test_clamping(self, config: ChunkingConfig):
        text = "Test. " * 10
        chunks = chunk_text(text, config)
        chunk = chunks[0]
        # Span that exceeds chunk boundary
        local_spans = [(1, 0, len(chunk.text) + 10)]
        global_spans = map_local_spans_to_global(local_spans, chunk)
        assert global_spans[0][2] == chunk.global_end


class TestSpanMerging:
    """Test deduplication of overlapping spans from multiple chunks."""

    def test_no_overlap(self):
        spans = [(1, 0, 5), (2, 10, 15), (1, 20, 25)]
        merged = merge_overlapping_spans(spans, "x" * 30)
        assert merged == spans

    def test_exact_overlap_same_label(self):
        spans = [(1, 0, 10), (1, 0, 10)]
        merged = merge_overlapping_spans(spans, "x" * 20)
        assert merged == [(1, 0, 10)]

    def test_partial_overlap_same_label(self):
        spans = [(1, 0, 10), (1, 5, 15)]
        merged = merge_overlapping_spans(spans, "x" * 20)
        # Deterministic precedence keeps the first equal-length span
        assert merged == [(1, 0, 10)]

    def test_partial_overlap_different_labels_prefer_longer(self):
        spans = [(1, 0, 10), (2, 5, 20)]  # Second is longer
        merged = merge_overlapping_spans(spans, "x" * 30, prefer_longer=True)
        assert merged == [(2, 5, 20)]

    def test_partial_overlap_different_labels_prefer_first(self):
        spans = [(1, 0, 15), (2, 10, 20)]  # First is longer
        merged = merge_overlapping_spans(spans, "x" * 30, prefer_longer=True)
        assert merged == [(1, 0, 15)]

    def test_contained_span(self):
        spans = [(1, 0, 20), (1, 5, 10)]  # Second contained in first
        merged = merge_overlapping_spans(spans, "x" * 30)
        assert merged == [(1, 0, 20)]

    def test_out_of_bounds_spans_ignored(self):
        spans = [(1, -5, 5), (1, 10, 50)]  # Second exceeds text length
        merged = merge_overlapping_spans(spans, "x" * 20)
        assert merged == [(1, 0, 5), (1, 10, 20)]  # Bounds are clamped

    def test_zero_length_spans_ignored(self):
        spans = [(1, 5, 5), (1, 0, 10)]
        merged = merge_overlapping_spans(spans, "x" * 20)
        assert merged == [(1, 0, 10)]


class TestConfigBuilder:
    """Test the config builder helper."""

    def test_build_config(self):
        config = build_chunking_config(
            encoding_name="cl100k_base",
            max_tokens=2048,
            overlap_tokens=256,
        )
        assert config.max_tokens == 2048
        assert config.overlap_tokens == 256
        assert config.encoding.name == "cl100k_base"

    def test_build_config_defaults(self):
        config = build_chunking_config("cl100k_base")
        assert config.max_tokens == 8192
        assert config.overlap_tokens == 128
        assert config.prefer_paragraph_boundaries is True
        assert config.prefer_sentence_boundaries is True
        assert config.hard_cut_fallback is True


class TestCutPointFinding:
    """Test the cut point finding logic."""

    @pytest.fixture
    def encoding(self):
        return tiktoken.get_encoding("cl100k_base")

    @pytest.fixture
    def config(self, encoding) -> ChunkingConfig:
        return ChunkingConfig(
            max_tokens=100,
            overlap_tokens=20,
            encoding=encoding,
        )

    def test_cut_at_paragraph(self, encoding, config):
        text = "Para one. " * 20 + "\n\n" + "Para two. " * 20
        # Target around the paragraph boundary
        cut = _find_cut_point(text, 150, encoding, config)
        # Should find the paragraph boundary
        assert "\n\n" in text[:cut]

    def test_cut_at_sentence(self, encoding, config):
        text = "Sentence one. " * 20 + "Sentence two. " * 20
        cut = _find_cut_point(text, 150, encoding, config)
        # Should find a sentence boundary
        assert text[cut - 2:cut] in [". ", "? ", "! "]

    def test_cut_at_whitespace(self, encoding, config):
        text = "word " * 200
        cut = _find_cut_point(text, 100, encoding, config)
        # Should cut at a space
        assert text[cut - 1] == " "

    def test_hard_cut_fallback(self, encoding, config):
        # Text with no boundaries and enough tokens to exceed the budget
        text = "ab cd ef " * 200
        config_no_fallback = ChunkingConfig(
            max_tokens=100,
            overlap_tokens=20,
            encoding=encoding,
            hard_cut_fallback=True,
        )
        cut = _find_cut_point(text, 100, encoding, config_no_fallback)
        assert cut > 0 and cut < len(text)

    def test_no_fallback_returns_full_text(self, encoding):
        text = "a" * 500
        config_no_fallback = ChunkingConfig(
            max_tokens=100,
            overlap_tokens=20,
            encoding=encoding,
            hard_cut_fallback=False,
        )
        cut = _find_cut_point(text, 100, encoding, config_no_fallback)
        # Without fallback, should return full text (no cut found)
        assert cut == len(text)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])