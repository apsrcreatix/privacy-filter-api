"""EXPLORATORY ONLY: superseded sentence-splitting experiment.

The supported implementation is tokenizer-aware ``opf._core.chunking``;
this file is retained only for comparison and is not part of the API.
"""

import re
import json
import os
from opf._api import OPF

# Force Triton off for Mac
os.environ['OPF_MOE_TRITON'] = '0'

# Initialize the OPF redactor
redactor = OPF(output_text_only=True)

def split_into_sentences(text: str) -> list[tuple[str, str]]:
    """
    Splits text into sentences/chunks, preserving the trailing whitespace/separators.
    Returns a list of tuples: (sentence_text, trailing_whitespace)
    """
    # Regex to split on sentence endings (. ? !) followed by space/newline,
    # avoiding common abbreviations like Mr., Ms., Dr., Dr, St., etc.
    sentence_pattern = re.compile(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)([\s\n]+)'
    )
    
    parts = sentence_pattern.split(text)
    chunks = []
    
    # The split result alternates between sentence content and separator
    # e.g., ["Sentence 1.", " ", "Sentence 2.", "\n", ...]
    for i in range(0, len(parts), 2):
        sentence = parts[i]
        if i + 1 < len(parts):
            separator = parts[i + 1]
        else:
            separator = ""
        if sentence or separator:
            chunks.append((sentence, separator))
            
    return chunks

def redact_paragraph(text: str) -> str:
    """
    Redacts a long paragraph by splitting it into sentences,
    redacting each sentence, and stitching them back together.
    """
    chunks = split_into_sentences(text)
    redacted_parts = []
    for sentence, separator in chunks:
        if sentence.strip():
            redacted_sentence = str(redactor.redact(sentence))
            redacted_parts.append(redacted_sentence + separator)
        else:
            redacted_parts.append(sentence + separator)
    return "".join(redacted_parts)

# Let's run a test on a paragraph that contains multiple PII points
test_para = (
    "My name is John Doe and I live at 123 Main Street in Atlanta, GA. "
    "You can reach me at john.doe@example.com or via my work number (404) 555-0199. "
    "I was born on July 4th, 1985 and my Social Security Number is 987-65-4321."
)

print("--- Original Text ---")
print(test_para)
print("\n--- Redacted with Sentence Splitting ---")
print(redact_paragraph(test_para))
