# Modality-Neutral Segment Contract for OPF Redaction

This document defines the canonical segment record used by OPF's document redaction API. The contract is modality-agnostic: the same record shape carries text from plain strings, OCR JSON, audio transcripts, subtitle cues, or caller-provided segment arrays.

## 1. Segment Record

A **segment** is a single, ordered piece of text with stable identity and optional metadata. The canonical Python type is a frozen dataclass; the JSON schema below is the wire format.

```python
@dataclass(frozen=True)
class Segment:
    text: str                           # The raw text content to redact
    source_id: str                      # Stable identifier for the source leaf (e.g., "ocr/0", "transcript/3")
    order: int                          # Global reading-order index (0-based, monotonically increasing)
    local_start: int = 0                # Offset of this segment's text within its source leaf (usually 0)
    local_end: int | None = None        # Exclusive end offset within the source leaf; None == len(text)
    metadata: dict[str, object] | None = None  # Opaque source-specific payload (boxes, timestamps, speaker, confidence...)
```

**Stability rules**

| Field            | Stability | Notes |
|------------------|-----------|-------|
| `text`           | Stable    | Redacted in place; original text preserved only in diagnostic output. |
| `source_id`      | Stable    | Must be unique within one request. Adapters generate stable IDs from source paths. |
| `order`          | Stable    | Monotonically increasing; defines the logical reading order for joining. |
| `local_start`/`local_end` | Stable | Offsets *inside the source leaf*, not the joined text. |
| `metadata`       | Opaque    | Preserved verbatim in output; core never inspects or mutates it. |

## 2. Logical Text Construction

The **logical text** presented to the model is the concatenation of all segment texts in `order` order, joined by a configurable separator (default: `"\n"`).

```python
def build_logical_text(segments: list[Segment], join_separator: str = "\n") -> tuple[str, list[tuple[int, int, str]]]:
    """
    Returns:
        (joined_text, mapping)
    where mapping is a list of (joined_start, joined_end, source_id) for each segment.
    """
    parts = []
    mapping = []
    cursor = 0
    for i, seg in enumerate(segments):
        start = cursor
        parts.append(seg.text)
        cursor += len(seg.text)
        end = cursor
        mapping.append((start, end, seg.source_id))
        if i < len(segments) - 1:
            parts.append(join_separator)
            cursor += len(join_separator)
    return join_separator.join(parts), mapping
```

The `mapping` array enables **span projection**: any span `[start, end)` in the joined text can be mapped back to the source segment(s) it touches.

## 3. Span Projection Policy

When the model predicts a span `[start, end)` in the joined logical text:

1. **Locate affected segments**: Find all segments whose `[joined_start, joined_end)` interval intersects the span.
2. **Single-segment span**: If the span lies entirely within one segment, replace `segment.text[local_start:local_end]` with the typed placeholder (e.g., `<PERSON>`).
3. **Cross-segment span**: If the span crosses segment boundaries:
   - Replace the covered portion in *each* affected segment with a **generic redaction marker** (default: `<REDACTED>`).
   - Emit a **diagnostic span record** in the report containing:
     - The full original joined-text span text
     - The typed label predicted by the model
     - The list of affected `(source_id, local_start, local_end)` tuples
   - Downstream consumers that need typed per-segment placeholders can reconstruct them from the diagnostic record.
4. **Never** invent a combined string that cannot be mapped back to source leaves.

## 4. Adapters (Input Kinds)

Callers provide input via one of the following `input_kind` values. Each adapter returns `(segments: list[Segment], warnings: list[str])`.

| `input_kind` | Description | Required fields |
|--------------|-------------|-----------------|
| `"text"` | Plain string | `input: str` |
| `"segments"` | Caller-provided segment array | `input: list[SegmentInput]` |
| `"ocr"` | Canonical PaddleOCR-like JSON | `input: dict`, `adapter: "paddleocr"` |
| `"transcript"` | Timestamped transcript segments | `input: list[TranscriptSegment]`, `adapter: "transcript"` |
| `"subtitle"` | WebVTT/SRT caption cues | `input: list[SubtitleCue]`, `adapter: "subtitle"` |
| `"json_paths"` | Arbitrary JSON with explicit text paths | `input: dict`, `text_paths: list[str]` |

### 4.1 `"text"` – Plain String
```json
{ "input": "Alice lives at 123 Main St.", "input_kind": "text" }
```
Produces one segment: `source_id="text/0", order=0, text="Alice lives at 123 Main St."`

### 4.2 `"segments"` – Explicit Segment Array
```json
{
  "input_kind": "segments",
  "input": [
    { "text": "Alice lives at", "source_id": "para/0", "order": 0 },
    { "text": "123 Main St.", "source_id": "para/1", "order": 1, "metadata": { "bbox": [10,20,100,30] } }
  ]
}
```
`SegmentInput` schema:
```json
{
  "text": "string",
  "source_id": "string",
  "order": "integer",
  "local_start": "integer (optional, default 0)",
  "local_end": "integer (optional)",
  "metadata": "object (optional)"
}
```

### 4.3 `"ocr"` – PaddleOCR / Canonical OCR
Canonical input shape (PaddleOCR `ocr_result`):
```json
{
  "input_kind": "ocr",
  "adapter": "paddleocr",
  "input": {
    "pages": [
      {
        "width": 1200, "height": 1600,
        "blocks": [
          { "text": "Alice", "bbox": [[10,20],[50,20],[50,40],[10,40]], "confidence": 0.99 },
          { "text": "123 Main St.", "bbox": [[10,50],[100,50],[100,70],[10,70]], "confidence": 0.95 }
        ]
      }
    ]
  }
}
```
Adapter behavior:
- Flattens `pages[i].blocks[j]` in reading order (top-to-bottom, left-to-right within line).
- `source_id = f"ocr/{page_idx}/{block_idx}"`
- `metadata = { "bbox": [...], "confidence": float, "page": int }`
- Warns if `blocks` array missing or `bbox`/`confidence` length mismatch.

### 4.4 `"transcript"` – Timestamped Audio Transcript
```json
{
  "input_kind": "transcript",
  "adapter": "transcript",
  "input": [
    { "text": "Hello, this is Alice.", "start": 0.0, "end": 2.5, "speaker": "A" },
    { "text": "My email is alice@example.com", "start": 2.5, "end": 5.0, "speaker": "A" }
  ]
}
```
Schema per segment:
```json
{ "text": "string", "start": "float", "end": "float", "speaker": "string (optional)", "confidence": "float (optional)" }
```
Adapter behavior:
- `source_id = f"transcript/{index}"`
- `metadata = { "start": float, "end": float, "speaker": str | None, "confidence": float | None }`
- Validates `start <= end` and monotonic non-decreasing `start` times; warns on violations.

### 4.5 `"subtitle"` – WebVTT / SRT Cues
```json
{
  "input_kind": "subtitle",
  "adapter": "subtitle",
  "input": [
    { "text": "Hello, this is Alice.", "start": "00:00:01.500", "end": "00:00:04.000", "id": "cue-1" },
    { "text": "My email is alice@example.com", "start": "00:00:04.500", "end": "00:00:07.000", "id": "cue-2" }
  ]
}
```
Schema per cue:
```json
{ "text": "string", "start": "string (HH:MM:SS.mmm)", "end": "string (HH:MM:SS.mmm)", "id": "string (optional)" }
```
Adapter behavior:
- Parses timestamps to float seconds.
- `source_id = f"subtitle/{id or index}"`
- `metadata = { "start_sec": float, "end_sec": float, "cue_id": str }`

### 4.6 `"json_paths"` – Arbitrary JSON with Explicit Paths
```json
{
  "input_kind": "json_paths",
  "input": {
    "document": {
      "pages": [
        { "text": "Alice lives at 123 Main St.", "meta": { "page": 1 } },
        { "text": "Email: alice@example.com", "meta": { "page": 2 } }
      ]
    }
  },
  "text_paths": [ "/document/pages/0/text", "/document/pages/1/text" ]
}
```
- `text_paths` are JSON Pointer paths (RFC 6901) or a simplified dot/bracket syntax.
- Each matched string leaf becomes one segment.
- `source_id = path` (the JSON pointer).
- `metadata = { "path": path, "parent": parent_object }` – the parent object is preserved so non-text siblings survive round-trip.
- Warns if a path resolves to a non-string or missing node.

## 5. Output Envelope

The document redaction API returns a canonical envelope that preserves the input kind and structure.

```json
{
  "schema_version": 2,
  "input_kind": "ocr",
  "adapter": "paddleocr",
  "redacted": { ... },              // Same shape as input, with segment.text replaced
  "report": {
    "schema_version": 2,
    "summary": { "span_count": 3, "by_label": { "PERSON": 1, "ADDRESS": 1, "EMAIL": 1 } },
    "spans": [                      // Diagnostic spans in logical-text coordinates
      { "label": "PERSON", "start": 0, "end": 5, "text": "Alice", "placeholder": "<PERSON>",
        "source_spans": [ { "source_id": "ocr/0/0", "local_start": 0, "local_end": 5 } ] },
      { "label": "ADDRESS", "start": 13, "end": 26, "text": "123 Main St.",
        "placeholder": "<ADDRESS>",
        "source_spans": [ { "source_id": "ocr/0/1", "local_start": 0, "local_end": 13 } ] }
    ],
    "warnings": [ "ocr: page 0 block 2 missing confidence" ],
    "decoded_mismatch": false
  }
}
```

### 5.1 Safe vs Diagnostic Output
- **`redacted`** (safe): Contains the input structure with `text` fields replaced by redacted text. Non-text fields (boxes, timestamps, speaker IDs, confidence scores) are preserved verbatim. No original PII text appears here.
- **`report`** (diagnostic): Contains original span text, exact logical-text offsets, typed placeholders, and source-span mappings. This object is **not** included in the safe output by default; callers must request `include_diagnostics=true` (or equivalent) to receive it.

## 6. Join Separator

The `join_separator` option (default: `"\n"`) controls how segment texts are concatenated for the model.

- Use `"\n"` for paragraphs, OCR blocks, transcript segments (preserves line breaks).
- Use `" "` for subtitle cues or word-level OCR where spaces are meaningful.
- Use `""` for character-level tasks (not recommended for PII).

The separator is **not** part of any segment's `text` and is never redacted.

## 7. Warnings

Warnings are non-fatal and collected in the report. Common warning codes:

| Code | Meaning |
|------|---------|
| `ocr.missing_confidence` | OCR block lacks confidence score |
| `ocr.mismatched_arrays` | `text`/`bbox`/`confidence` length mismatch |
| `transcript.non_monotonic_time` | Segment start times not non-decreasing |
| `transcript.overlap` | Segment time ranges overlap |
| `subtitle.timestamp_parse` | Failed to parse HH:MM:SS.mmm |
| `json_paths.not_found` | JSON pointer did not resolve |
| `json_paths.non_string_leaf` | Resolved leaf is not a string |
| `chunking.tokenizer_mismatch` | Tokenizer decode ≠ original text (spans based on decoded) |
| `projection.cross_segment_span` | Model span crossed segment boundary (generic marker used) |
| `projection.trimmed_whitespace` | Leading/trailing whitespace trimmed from span |

## 8. Schema Versioning

- `schema_version` in the top-level envelope increments only for **breaking** changes to the envelope or segment contract.
- `report.schema_version` tracks the diagnostic report format.
- Adapters are versioned independently (e.g., `adapter: "paddleocr@v1"`). Breaking adapter changes get a new version suffix.

## 9. Examples

### 9.1 PaddleOCR with Cross-Box Person Name
Input (two OCR boxes: `"John"` and `"Doe"`):
```json
{
  "input_kind": "ocr",
  "adapter": "paddleocr",
  "input": { "pages": [{ "blocks": [
    { "text": "John", "bbox": [[10,10],[40,10],[40,30],[10,30]], "confidence": 0.99 },
    { "text": "Doe", "bbox": [[50,10],[80,10],[80,30],[50,30]], "confidence": 0.98 }
  ]}] }
}
```
Joined text (with `"\n"` separator): `"John\nDoe"`
Model predicts span `[0, 8)` → `"John\nDoe"` (label `PERSON`).

**Projection**: Crosses segment boundary.
- `redacted`: Both boxes get `"<REDACTED>"` in their `text` field.
- `report.spans`: One diagnostic span with `source_spans` listing both boxes.

### 9.2 Transcript with PII Split Across Segments
```json
{
  "input_kind": "transcript",
  "adapter": "transcript",
  "input": [
    { "text": "My email is alice@", "start": 0.0, "end": 2.0 },
    { "text": "example.com", "start": 2.0, "end": 3.0 }
  ]
}
```
Joined: `"My email is alice@\nexample.com"`
Model span `[12, 26)` → `"alice@\nexample.com"` (label `EMAIL`).

**Projection**: Crosses boundary.
- `redacted`: Segment 0 text → `"My email is <REDACTED>"`, segment 1 text → `"<REDACTED>"`
- `report`: Single diagnostic span with both `source_spans`.

### 9.3 Plain Text (Backward-Compatible)
```json
{ "input": "Alice at alice@example.com", "input_kind": "text" }
```
Output envelope uses `input_kind: "text"`, `redacted` is a string, `report.spans` uses `source_id: "text/0"`.

## 10. Out of Scope

- **Arbitrary recursive JSON redaction**: The `"json_paths"` adapter requires explicit paths. A recursive "redact every string" mode may exist as a convenience (`input_kind: "json_auto"`) but **must be explicitly opted into** and is labelled *lossy/risky* in the response warnings.
- **Audio/video pixel redaction**: This contract only covers text and metadata. Actual waveform muting, beep insertion, or video box blurring are downstream operations that consume the `report.spans` intervals.
- **Semantic segmentation**: The segment contract carries whatever boundaries the adapter provides. It does not impose sentence/paragraph semantics; the chunking layer (Task 2) handles model-window chunking independently.