# OPF Output Schemas

## 1. `opf` / `opf redact` JSON Output

Printed to stdout once per input example.

```json
{
  "schema_version": 1,
  "summary": {
    "output_mode": "typed",
    "span_count": 3,
    "by_label": {
      "private_person": 1,
      "private_date": 2
    },
    "decoded_mismatch": false
  },
  "text": "Alice was born on 1990-01-02.",
  "detected_spans": [
    {
      "label": "private_person",
      "start": 0,
      "end": 5,
      "text": "Alice",
      "placeholder": "<PRIVATE_PERSON>"
    }
  ],
  "redacted_text": "<PRIVATE_PERSON> was born on <PRIVATE_DATE>."
}
```

Notes:

- In `--output-mode redacted`, every `detected_spans[*].label` becomes `redacted`.
- `warning` is present only when tokenizer decode does not exactly round-trip the input text.

## 2. `opf eval` Predictions Output (`--predictions-out`)

Written as JSONL when requested.

```json
{
  "example_id": "stable-id",
  "text": "Alice was born on 1990-01-02.",
  "predicted_spans": {
    "private_person: Alice": [[0, 5]]
  }
}
```

Optional field:

- `token_logprobs_topk`: included only when `--predictions-token-logprobs-topk > 0`

Notes:

- This file is literal JSONL: one compact JSON object per line.

## 3. Document/Structured Redaction API (schema_version: 2)

The document redaction endpoint accepts a canonical request envelope and returns a structured response that preserves the input shape while providing a safe redacted payload and an optional diagnostic report.

### 3.1 Request Envelope

```json
{
  "input": {},
  "input_kind": "text|segments|ocr|transcript|subtitle|json_paths|json_auto",
  "adapter": "optional-adapter-name",
  "text_paths": ["/pages/0/text"],
  "options": {
    "join_separator": "\n",
    "output_mode": "typed",
    "include_diagnostics": false,
    "chunk_overlap_tokens": 128,
    "max_chunk_tokens": 8192
  }
}
```

**Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `input` | any | yes | — | The payload to redact. Shape depends on `input_kind`. |
| `input_kind` | string | yes | — | One of: `text`, `segments`, `ocr`, `transcript`, `subtitle`, `json_paths`, `json_auto`. |
| `adapter` | string | no | inferred from `input_kind` | Explicit adapter name (e.g., `paddleocr@v1`, `transcript@v1`). |
| `text_paths` | array[string] | conditional | — | Required when `input_kind == "json_paths"`. JSON Pointer paths to redactable text leaves. |
| `options.join_separator` | string | no | `"\n"` | Separator inserted between segments when joining for model input. |
| `options.output_mode` | string | no | `"typed"` | `"typed"` (preserve labels) or `"redacted"` (collapse to generic). |
| `options.include_diagnostics` | boolean | no | `false` | If `true`, include full diagnostic `report` in response. |
| `options.chunk_overlap_tokens` | int | no | `128` | Token overlap between chunks for long-text processing. |
| `options.max_chunk_tokens` | int | no | `8192` | Max tokens per chunk (bounded by model context). |

### 3.2 Response Envelope

```json
{
  "schema_version": 2,
  "input_kind": "ocr",
  "adapter": "paddleocr@v1",
  "redacted": { ... },
  "report": {
    "schema_version": 2,
    "summary": {
      "output_mode": "typed",
      "span_count": 3,
      "by_label": { "private_person": 1, "private_address": 1, "private_email": 1 },
      "decoded_mismatch": false
    },
    "spans": [
      {
        "label": "private_person",
        "start": 0,
        "end": 5,
        "text": "Alice",
        "placeholder": "<PRIVATE_PERSON>",
        "source_spans": [
          { "source_id": "ocr/0/0", "local_start": 0, "local_end": 5 }
        ]
      },
      {
        "label": "private_email",
        "start": 20,
        "end": 38,
        "text": "alice@example.com",
        "placeholder": "<PRIVATE_EMAIL>",
        "source_spans": [
          { "source_id": "transcript/1", "local_start": 12, "local_end": 30 }
        ]
      }
    ],
    "warnings": [
      "projection.cross_segment_span: span [20,38) crosses transcript/0 and transcript/1"
    ],
    "decoded_mismatch": false
  }
}
```

**Top-level fields:**

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Envelope schema version (2 for document API). |
| `input_kind` | string | Echo of request `input_kind`. |
| `adapter` | string | Adapter used (with version suffix). |
| `redacted` | any | **Safe output** — same shape as `input`, with `text` fields replaced by redacted text. Non-text fields (boxes, timestamps, speaker IDs, confidence, arbitrary metadata) preserved verbatim. |
| `report` | object | **Diagnostic output** — only present if `include_diagnostics=true`. Contains summary, typed spans in logical-text coordinates, source-span mappings, warnings, and tokenizer mismatch flag. |

### 3.3 Redacted Payload Shapes by Input Kind

| `input_kind` | `redacted` type | Notes |
|--------------|-----------------|-------|
| `text` | string | Redacted plain text. |
| `segments` | array[SegmentOutput] | Each segment: `{ "text": "...", "source_id": "...", "order": 0, "metadata": {...} }` |
| `ocr` | object | Mirrors input OCR structure; `blocks[i].text` replaced. |
| `transcript` | array[TranscriptOutput] | Each: `{ "text": "...", "start": float, "end": float, "speaker": "...", ... }` |
| `subtitle` | array[SubtitleOutput] | Each: `{ "text": "...", "start": "...", "end": "...", "id": "..." }` |
| `json_paths` | object | Input JSON with only the targeted string leaves replaced. |
| `json_auto` | object | **Lossy/risky** — recursive string replacement; original shape preserved but no span mapping. Warning emitted. |

### 3.4 Span Record (Diagnostic Report)

Each span in `report.spans`:

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | Model label (e.g., `private_person`) or `redacted` if `output_mode=redacted`. |
| `start` | int | Start offset in **joined logical text** (0-indexed, character). |
| `end` | int | End offset (exclusive) in joined logical text. |
| `text` | string | Original span text from joined logical text. |
| `placeholder` | string | Placeholder used in `redacted` (e.g., `<PRIVATE_PERSON>` or `<REDACTED>`). |
| `source_spans` | array[SourceSpan] | Projection back to source segments. |

**SourceSpan:**

| Field | Type | Description |
|-------|------|-------------|
| `source_id` | string | Segment `source_id` (e.g., `ocr/0/0`, `transcript/1`, `text/0`). |
| `local_start` | int | Start offset within that segment's `text`. |
| `local_end` | int | End offset (exclusive) within that segment's `text`. |

### 3.5 Warnings

Collected in `report.warnings` (array of strings). Standard codes:

| Code | Meaning |
|------|---------|
| `ocr.missing_confidence` | OCR block lacks confidence score. |
| `ocr.mismatched_arrays` | `text`/`bbox`/`confidence` length mismatch. |
| `transcript.non_monotonic_time` | Segment `start` times not non-decreasing. |
| `transcript.overlap` | Segment time ranges overlap. |
| `subtitle.timestamp_parse` | Failed to parse HH:MM:SS.mmm timestamp. |
| `json_paths.not_found` | JSON Pointer did not resolve. |
| `json_paths.non_string_leaf` | Resolved leaf is not a string. |
| `chunking.tokenizer_mismatch` | Tokenizer decode ≠ original text; spans based on decoded. |
| `projection.cross_segment_span` | Model span crossed segment boundary; generic marker used in `redacted`. |
| `projection.trimmed_whitespace` | Leading/trailing whitespace trimmed from detected span. |
| `json_auto.lossy_redaction` | `json_auto` mode used; no span mapping available. |

---

## Stability Notes

- `typed` and `untyped` are the evaluation terms.
- `typed` and `redacted` are the prediction-output terms.
- Additive fields may appear over time, but existing keys should remain stable unless `schema_version` changes for API/CLI JSON payloads.
