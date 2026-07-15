"""EXPLORATORY ONLY: legacy FastAPI prototype.

This module is not the supported document API. Use ``OPF.redact_document``
with explicit adapters from ``opf._core.documents`` instead. It recursively
redacts JSON strings and initializes a model at import time, so it is retained
only as a historical prototype and must not be used as a production contract.
"""

import os
import re
import json
import asyncio
from typing import Optional, Any, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Force Triton off for macOS
os.environ['OPF_MOE_TRITON'] = '0'

from opf._api import OPF, RedactionResult

# Initialize a single OPF redactor (shared across background worker tasks)
redactor = OPF(output_text_only=True)

# ----------------------------------------------------------------------
# Smart NLP/Regex Sentence Splitter for Paragraphs
# ----------------------------------------------------------------------
def split_into_sentences(text: str) -> list[tuple[str, str]]:
    """
    Splits text into sentences/chunks, preserving the trailing whitespace/separators.
    Returns a list of tuples: (sentence_text, trailing_whitespace)
    """
    # Regex splits on sentence endings (. ? !) followed by space/newline,
    # avoiding common abbreviations like Mr., Ms., Dr., St., etc.
    sentence_pattern = re.compile(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)([\s\n]+)'
    )
    
    parts = sentence_pattern.split(text)
    chunks = []
    
    for i in range(0, len(parts), 2):
        sentence = parts[i]
        separator = parts[i + 1] if i + 1 < len(parts) else ""
        if sentence or separator:
            chunks.append((sentence, separator))
            
    return chunks

def redact_sentence_by_sentence(text: str) -> str:
    """
    Redacts any raw text by breaking it into sentences, redacting each 
    individually, and stitching them back together to preserve spacing.
    """
    if not text or not text.strip():
        return text
    
    chunks = split_into_sentences(text)
    redacted_parts = []
    for sentence, separator in chunks:
        if sentence.strip():
            # Call OPF redactor on the individual sentence
            redacted_sentence = str(redactor.redact(sentence))
            redacted_parts.append(redacted_sentence + separator)
        else:
            redacted_parts.append(sentence + separator)
            
    return "".join(redacted_parts)

# ----------------------------------------------------------------------
# Recursive JSON Layout-Preserving Traverser (for OCR, nested Dicts/Lists)
# ----------------------------------------------------------------------
def recursive_redact(data: Any) -> Any:
    """
    Recursively traverses dictionaries, lists, and values.
    Preserves all JSON structural elements (keys, coordinates, bounding boxes)
    while applying smart sentence-based redaction to values.
    """
    if isinstance(data, str):
        # Apply the smart sentence redactor to raw strings
        return redact_sentence_by_sentence(data)
    
    elif isinstance(data, dict):
        # Traverse dictionary keys and redact only their values
        # This keeps structural information like "compliance_scope", "box", etc., completely clean
        return {key: recursive_redact(value) for key, value in data.items()}
    
    elif isinstance(data, list):
        # Traverse list items (such as OCR bounding box arrays or lists of records)
        return [recursive_redact(item) for item in data]
    
    # Return non-string leaf nodes (numbers, booleans, nulls) as-is
    return data

# ----------------------------------------------------------------------
# FastAPI Setup with Asyncio Queue
# ----------------------------------------------------------------------
app = FastAPI(
    title="OPF Enterprise Redaction API",
    description="Privacy redaction service supporting unstructured text, key-value JSON, and PaddleOCR layout structures.",
    version="1.0.0"
)

class RedactRequest(BaseModel):
    # Accepting any valid JSON value: string, dict, or list
    data: Any = Field(
        ..., 
        example={
            "text_content": "Dear John Doe, please call me back.",
            "ocr_blocks": [
                {"text": "John Doe", "box": [10, 20, 100, 30]},
                {"text": "123 Main St", "box": [10, 40, 100, 50]}
            ]
        }
    )

class RedactResponse(BaseModel):
    redacted_data: Any

MAX_QUEUE_SIZE = 100
request_queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)

async def worker() -> None:
    """Background task pulling requests from the queue to process them in executor threads."""
    while True:
        future: asyncio.Future
        req_data: Any
        future, req_data = await request_queue.get()
        try:
            loop = asyncio.get_running_loop()
            # Run blocking redaction execution in the thread pool
            result = await loop.run_in_executor(
                None,
                lambda: recursive_redact(req_data)
            )
            if not future.done():
                future.set_result(result)
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
        finally:
            request_queue.task_done()

@app.on_event("startup")
async def startup_event():
    # Start the processing worker loop
    asyncio.create_task(worker())
    print("OPF enterprise worker active - queue initialized.")

@app.post("/redact", response_model=RedactResponse)
async def redact_endpoint(req: RedactRequest):
    """
    Accepts any arbitrary JSON structure (unstructured text, records, or OCR outputs),
    preserves layout/coordinates/structures, and redacts sensitive data inside string values.
    """
    if request_queue.qsize() >= MAX_QUEUE_SIZE:
        raise HTTPException(status_code=429, detail="Queue capacity reached. Please try again.")

    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    await request_queue.put((future, req.data))

    try:
        redacted_result = await future
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return RedactResponse(redacted_data=redacted_result)

@app.get("/healthz")
async def healthz():
    return {"status": "healthy", "queued_items": request_queue.qsize()}
