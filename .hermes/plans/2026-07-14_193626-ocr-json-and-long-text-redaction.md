# OCR JSON and Long-Text Redaction Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a stable, input-agnostic text-segment interface that can redact text from JSON/OCR, audio transcripts, subtitles, document parsers, and long paragraphs while preserving source structure and timing/layout metadata.

**Architecture:** Keep `OPF.redact(str)` as the primitive. Add a modality-neutral segment layer whose records contain text, source identity/path, local offsets, ordering, and optional timing/layout metadata. Adapters convert OCR JSON, transcript segments, subtitle cues, or user-provided arrays into that layer; the core redacts joined logical text and projects replacements back without knowing the source modality. For long text, use a dependency-light boundary-aware chunker with token-budget limits and overlap; retain the existing model-window path as a fallback, but make cross-chunk span merging explicit. Add optional third-party segmenters only behind an extra dependency rather than making a heavyweight NLP model mandatory.

**Tech Stack:** Existing Python 3.10+ package, OPF tokenizer/runtime, standard-library JSON/path traversal, pytest-style tests. Optional segmenter support, if later justified by benchmarks, should be an extra rather than a core dependency.

---

## Findings and design decision

- `OPF.redact` already returns character-offset spans for a single string and is the correct primitive to preserve.
- The exploratory `api_server.py` recursively redacts every string and sentence-splits with regex. It is not yet a safe document contract: it can redact metadata that is not OCR text, loses span metadata, makes one model call per sentence, and cannot reliably handle PII crossing sentence/OCR-item boundaries.
- `predict_text` currently creates fixed, non-overlapping token windows. A sentence splitter may reduce individual input size but does not solve entities crossing model-window or sentence boundaries.
- OCR output is not one universal JSON shape, and OCR is only one producer. We should support a modality-neutral segment contract plus explicit adapters for canonical OCR, timestamped audio transcripts, subtitles, and user-supplied segment arrays. Do not promise that arbitrary recursive JSON can distinguish text from filenames, labels, URLs, or metadata.
- The safest initial segmentation strategy is layered: preserve paragraph/newline boundaries, identify likely sentence boundaries with a robust standard-library heuristic, enforce a token budget, and add overlap/context at chunk boundaries. Measure before selecting a third-party NLP segmenter.

## Proposed public contracts

1. **Text mode:** Existing `OPF.redact(text)` remains backward-compatible.
2. **Document mode:** A new document function/API accepts JSON-compatible values and an explicit mode/schema, rather than silently treating every string leaf as OCR text.
3. **Segment mode:** Accept ordered text segments with stable source IDs, optional separators, and optional metadata such as OCR boxes or audio `start`/`end` timestamps. Redact the logical joined text, then project each span back to the source segments.
4. **Adapter modes:** Provide explicit adapters for canonical OCR JSON, timestamped transcript JSON, subtitle cues, and allowlisted JSON paths. Adapters preserve source-specific metadata but the core sees only segments. A recursive “redact every string” mode may exist for convenience but must be explicitly labelled lossy/risky.
5. **Safe versus diagnostic output:** The safe response should contain redacted source data plus non-sensitive path/label/count metadata by default. Original source text, matched span text, detailed offsets, and model diagnostics belong in a separately authorized diagnostic/audit result, because the existing single-string `RedactionResult` intentionally exposes source text for inspection but should not automatically be embedded in an untrusted redaction response.

## Server boundary: flexible transport, explicit semantics

The server should accept multiple transport forms but use one canonical request envelope internally. Transport and semantic interpretation are separate:

- `text/plain` may be accepted as a convenience and wrapped as `input_kind="text"`.
- `application/json` should accept the envelope and arbitrary JSON payloads.
- JSONL, multipart, and binary/audio uploads can be added later through separate adapters; they should not change the core segment contract.

Do not make `any JSON` mean “redact every string.” Any valid JSON can be preserved structurally, but the server cannot safely infer which strings are text, their reading order, or whether a string is metadata. Require one of `text`, `segments`, a named adapter, or explicit JSON paths. An optional `auto` mode may recognize known schemas, but ambiguous inputs should return a validation response rather than guess.

Conceptually, the canonical request is:

```json
{
  "input": {},
  "input_kind": "text|segments|json",
  "adapter": "optional-adapter-name",
  "text_paths": ["/pages/0/text"],
  "options": {
    "join_separator": "\\n",
    "output_mode": "redacted",
    "include_diagnostics": false
  }
}
```

The response should preserve the input kind and JSON shape, return the redacted payload, and include versioned warnings/report metadata. This accommodates different callers without making every caller implement a custom endpoint.

## Span projection policy

- Build a mapping from logical joined-text offsets to `(source ID/path, local start, local end)` and retain opaque source metadata separately.
- For spans wholly inside one leaf, use the typed placeholder from OPF.
- For spans crossing leaves, mark all covered leaves and use a consistent cross-segment replacement policy (initial recommendation: replace the covered portion in each leaf with a generic redaction marker, with the full typed span retained in the report). Never silently invent a combined OCR string that cannot be mapped back.
- Preserve array order, dictionary keys, coordinate arrays, confidence values, timestamps, speaker IDs, non-string values, and original whitespace unless the caller selects a normalization mode. Text redaction must not imply audio/image pixel redaction.
- Include a schema version and warnings for ambiguous/malformed OCR structures, tokenizer round-trip mismatches, and cross-leaf spans.

## Long-paragraph strategy

- First establish a token-budget chunking path based on the model’s actual tokenizer and configured runtime context. Do not use character counts as a proxy.
- Prefer cuts after paragraph/newline boundaries, then sentence boundaries, then safe whitespace, and finally hard token cuts for pathological/OCR text.
- Add configurable overlap/context around each chunk. Overlap should be large enough for names/addresses and configurable in tokens or sentences.
- Convert each chunk’s local spans into global offsets, merge duplicate/overlapping predictions deterministically, and apply redaction once to the original text. Do not concatenate independently redacted chunks because that can duplicate placeholders or corrupt whitespace.
- Keep a bounded fallback for very long unbroken strings and noisy OCR where sentence detection is unreliable.
- Evaluate whether the existing runtime should itself change from non-overlapping windows to overlapping windows with aggregation; this is likely a better core fix than adding a paragraph NLP dependency.

## Open-source segmenter decision

Do not add spaCy, a statistical Punkt model, or a semantic chunking model in the first implementation. They increase install size, model/data handling, language/licensing surface, and can behave poorly on OCR noise. Use a tested, dependency-light heuristic plus tokenizer-aware boundaries first.

Consider an optional extra only after benchmark results:

- `syntok`/`blingfire`: attractive for fast sentence boundaries, but verify package maintenance, wheels, language behavior, and licensing before adoption.
- spaCy `Sentencizer`: practical and rule-based, but heavyweight for this package and still not a privacy-boundary solution.
- Punkt: useful for ordinary prose but abbreviation/model handling and OCR robustness need validation.
- `wtpsplit` or semantic chunkers: likely overkill for the initial redaction path and may introduce model downloads/latency.

The implementation should isolate the boundary detector behind a small interface so a later optional segmenter can be benchmarked without changing JSON projection or redaction semantics.

---

## Implementation tasks

### Task 1: Define the modality-neutral segment contract

**Files:**
- Modify: `OUTPUT_SCHEMAS.md`
- Modify: `README.md`
- Create: `docs/SEGMENT_INPUTS.md` or an equivalent focused document

Document the segment record and logical-text mapping first. Include examples for sequential strings, PaddleOCR-like OCR records, timestamped audio transcripts, subtitle cues, explicit text-path selection, join separators, cross-segment spans, unchanged metadata, warnings, and out-of-scope arbitrary JSON semantics.

**Verification:** Review examples against the actual PaddleOCR payload variants to be supported; ensure every returned field has a documented stability rule.

### Task 2: Extract a reusable text segmentation/chunking module

**Files:**
- Create: `opf/_core/chunking.py`
- Test: `tests/test_chunking.py` (or the repository’s established test location)

Implement boundary candidates, tokenizer-aware token budgets, fallback hard cuts, stable global offsets, and configurable overlap. Keep the public surface independent of any optional third-party NLP package and independent of OCR/audio concepts.

**Tests:** prose with abbreviations; newlines; punctuation; no-punctuation OCR; Unicode; empty text; exact-boundary text; overlong single token/string; overlap coverage; global offset correctness.

### Task 3: Add overlap-aware long-text prediction and span merging

**Files:**
- Modify: `opf/_core/runtime.py`
- Possibly modify: `opf/_core/sequence_labeling.py`
- Test: runtime-focused tests using mocked model/logits where feasible

Decide whether to reuse `example_to_windows` with overlap or add a higher-level chunk path. Convert local spans to global offsets, deduplicate overlaps, retain deterministic label/replacement precedence, and perform one final redaction against the original source text.

**Tests:** PII at chunk edges, PII crossing an edge, duplicate detections in overlap, no span duplication, whitespace preservation, tokenizer decode mismatch handling, CPU context override.

### Task 4: Introduce normalized document segments

**Files:**
- Create: `opf/_core/documents.py` (or a similarly named module)
- Test: `tests/test_documents.py`

- Implement explicit adapters for:

- a list/array of text strings;
- a canonical PaddleOCR representation with text arrays and parallel metadata arrays;
- timestamped transcript segments such as `{text, start, end, speaker}`;
- subtitle/caption cues with timing and cue identifiers;
- nested records with an explicit text-field/path allowlist.

Reject or warn on mismatched parallel array lengths and ambiguous structures; never mutate keys or non-text values.

### Task 5: Add document redaction API and serialization

**Files:**
- Modify: `opf/_api.py`
- Modify: `OUTPUT_SCHEMAS.md`
- Test: `tests/test_api_documents.py`

Add a structured segment/document result that includes redacted data, source-level reports, schema version, and warnings. For timed audio/transcript inputs, preserve timestamps and speaker metadata and report redaction intervals in both source-segment and logical-text coordinates. Reuse the existing typed span model where possible. Preserve `OPF.redact(str)` behavior and avoid changing its return type.

**Tests:** round-trip JSON shape; nested paths; empty/null/numeric leaves; explicit field selection; cross-segment span; typed versus generic redaction; serialization with Unicode; warning propagation.

### Task 6: Add a thin CLI/API integration layer

**Files:**
- Create or modify: `opf/_cli` input/output modules
- Treat: `api_server.py` as exploratory until its contract is migrated
- Test: CLI/API integration tests

Support `text/plain` convenience input plus a canonical `application/json` request envelope. Accept arbitrary JSON as a preserved payload only when an adapter, `input_kind`, or explicit text paths identify the redactable text. Do not initialize a global model at import time; make device, checkpoint, context, and optional Triton settings configurable. Keep queueing/server concerns separate from document normalization and adapter selection.

**Tests:** JSONL one-object-per-line behavior, stdin/stdout separation, malformed JSON errors, queue saturation response, health endpoint, and no redaction of non-text OCR metadata.

### Task 7: Benchmark segmentation choices before adding dependencies

**Files:**
- Create: `tests/fixtures/` golden OCR/paragraph fixtures
- Create: `benchmarks/` or documented local benchmark script

Build a small synthetic and hand-reviewed corpus covering names, dates, emails, phones, addresses, secrets, OCR spacing errors, multilingual samples, and boundary-crossing PII. Compare: current behavior, regex sentence splitting, tokenizer-aware chunking with overlap, and any optional segmenter candidate.

Track span recall, false-positive rate, exact redacted-text preservation, cross-boundary recall, latency, peak memory, and package/install cost. Choose a third-party segmenter only if it improves privacy-relevant metrics materially without unacceptable deployment cost.

### Task 8: Remove or quarantine exploratory code

**Files:**
- `api_server.py`
- `test_sentence_splitter.py`

Either migrate useful behavior into tested package modules and delete these untracked experiments, or clearly mark them as examples that do not define the supported API. Do not leave duplicate sentence-splitting implementations with divergent behavior.

### Task 9: Verify the complete path

Run the project’s test suite plus focused document/chunking tests, static checks if configured, and a real local inference smoke test on:

- one paragraph containing multiple PII classes;
- an entity crossing a chunk boundary;
- a list of OCR strings with boxes;
- a canonical PaddleOCR-like payload;
- timestamped transcript segments with PII split across segments;
- subtitle cues with speaker/timing metadata;
- malformed and metadata-heavy JSON.

Confirm exact JSON round-trip for non-text fields, stable offsets, no duplicate placeholders, and expected CPU/MPS/device behavior.

---

## Risks and tradeoffs

- **Cross-leaf replacement ambiguity:** There is no universally correct visual redaction when a PII span spans two OCR boxes. Preserve a report and make the replacement policy explicit.
- **Producer/schema drift:** OCR, transcript, subtitle, and document tools all emit different shapes. Versioned adapters and fixtures are safer than broad recursive guessing.
- **Temporal redaction semantics:** Replacing transcript text does not alter the original audio. Audio waveform redaction, beep insertion, silence, and video subtitle burn-in are separate downstream operations that may consume the redaction intervals.
- **Format-versus-semantics confusion:** Supporting arbitrary JSON transport does not mean arbitrary JSON is self-describing. Preserve unknown fields, but fail closed when text locations or ordering are ambiguous.
- **Model-boundary errors:** More chunking can improve recall but may reduce context and increase latency. Overlap plus global merge is the key safeguard.
- **Sentence segmentation errors:** No segmenter is reliable on all languages, abbreviations, or OCR artifacts. It must be a hint for chunk boundaries, not a privacy decision-maker.
- **Concurrency:** A shared PyTorch model and thread executor may not be safe or performant under arbitrary parallelism. The server queue should have explicit worker/device policy and load testing.
- **Schema compatibility:** Add fields rather than changing existing `RedactionResult`; increment schema versions only for breaking changes.

## Questions to settle before implementation

1. What is the minimum neutral segment contract: text, source ID, ordering, local offsets, optional metadata, and optional time interval?
2. Which first adapters matter most: OCR, transcript JSON, WebVTT/SRT, or plain segment arrays?
3. Should the server accept `text/plain` convenience input in addition to the canonical JSON envelope?
4. For a span spanning adjacent segments, should each affected segment receive a generic marker while downstream consumers receive a full redaction interval?
5. Should JSON/segment mode default to typed placeholders or generic redaction?
6. Is the primary deployment a library, CLI, FastAPI service, or all three?
7. What languages, transcription errors, OCR errors, and timestamp granularity are required for the first benchmark?
8. Should long-text processing preserve a single global report with offsets, or return per-chunk diagnostics as well?

## Recommended implementation order

1. Lock schemas and fixtures.
2. Fix tokenizer-aware overlapping long-text inference and global span merging.
3. Add normalized modality-neutral segments and projection.
4. Add explicit OCR, transcript, and subtitle adapters.
5. Add CLI/server integration.
6. Benchmark optional sentence segmenters; add one if data supports it.
