"""Transcript analysis as the evaluation runs it: the app's own functions.

Since B6 this is a thin timing wrapper over ``streamlit_app``'s
``prefill_transcript``, ``summarize_transcript`` and ``score_question``, so
what is scored is exactly what ships — prompt, cache handling, retry and
all. B3's own copy of that logic, and its ``plan``/``anyone`` prompt
variants, are in git history (4eb24d3); the tuning they did is recorded in
CLAUDE.md.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any

from streamlit_app import (
    LLM_ID,
    LLM_REVISION,
    Summary,
    _yes_no_ids,
    prefill_transcript,
    score_question,
    summarize_transcript,
)

# The shipped model, and the 8B that B4 rejected, kept for comparisons (it is
# no longer in the local cache, so `--model 8b` downloads 4.95 GB).
MODELS: dict[str, tuple[str, str]] = {
    "3b": (LLM_ID, LLM_REVISION),
    "8b": (
        "ibm-granite/granite-4.2-8b-q4-mlx",
        "9047f073e46c527f70dade07bb55b1700d728e36",
    ),
}


@dataclass
class ExcerptResult:
    prefix_tokens: int
    prefill_seconds: float
    summary_seconds: float
    questions_seconds: float
    # The summary as JSON, or the RuntimeError's message when, after the
    # shipped retry, there was none.
    summary_text: str
    summary: Summary | None
    p_yes: dict[str, float] = field(default_factory=dict)
    yes_no_mass: dict[str, float] = field(default_factory=dict)


def analyse(
    model: Any, tokenizer: Any, transcript: str, questions: dict[str, str]
) -> ExcerptResult:
    began = time.perf_counter()
    prefill = prefill_transcript(model, tokenizer, transcript)
    prefill_seconds = time.perf_counter() - began

    began = time.perf_counter()
    summary: Summary | None
    try:
        summary = summarize_transcript(model, tokenizer, prefill)
        summary_text = json.dumps(summary)
    except RuntimeError as e:
        summary, summary_text = None, str(e)
    summary_seconds = time.perf_counter() - began

    result = ExcerptResult(
        prefix_tokens=len(prefill.ids),
        prefill_seconds=prefill_seconds,
        summary_seconds=summary_seconds,
        questions_seconds=0.0,
        summary_text=summary_text,
        summary=summary,
    )
    began = time.perf_counter()
    answer_ids = _yes_no_ids(tokenizer)
    for key, question in questions.items():
        p_yes, mass = score_question(model, tokenizer, prefill, question, answer_ids)
        result.p_yes[key], result.yes_no_mass[key] = p_yes, mass
    result.questions_seconds = time.perf_counter() - began
    assert prefill.cache[0].offset == len(prefill.ids)
    return result
