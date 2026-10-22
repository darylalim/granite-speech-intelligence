"""Transcript analysis as the evaluation runs it, ahead of the app code.

This is PLAN.md's Phase 1 and 2 design — one shared transcript prefill, a
greedy JSON summary that is trimmed off the cache, then one short forward pass
per question read out as P(yes) = P(Yes) / (P(Yes) + P(No)) — written here so
B3 can score it before B5 and B6 put it in ``streamlit_app.py``. Once they do,
the evaluation should import the shipped functions instead and this module
should go, so that what is scored is what ships.
"""

import json
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler

from streamlit_app import (
    ANALYSIS_SYSTEM_PROMPT,
    ANALYSIS_TRANSCRIPT_PROMPT,
    SUMMARY_PROMPT,
)

# The two candidates (PLAN.md, Model): the 8B is the plan's choice, the 3B the
# fallback B3 scores because the 8B missed the full-analysis budget in B2.
# Each is pinned to a full commit hash, for the reason MODEL_REVISION is.
MODELS: dict[str, tuple[str, str]] = {
    "8b": (
        "ibm-granite/granite-4.2-8b-q4-mlx",
        "9047f073e46c527f70dade07bb55b1700d728e36",
    ),
    "3b": (
        "ibm-granite/granite-4.2-3b-q8-mlx",
        "8e19ac6574124635eb8fbd06a15d05ebbfff82dd",
    ),
}
PREFILL_STEP = 2048
SUMMARY_MAX_TOKENS = 512


@dataclass(frozen=True)
class Prompt:
    """One wording of the prompt. Variants are compared on the tuning half only."""

    system: str
    transcript: str
    question: str
    summary: str


# PLAN.md's wording, as written in Phase 1 and 2.
_PLAN = Prompt(
    # The system, transcript and summary turns are the app's own since B5, so
    # what is scored is what ships; only the question wording lives here until
    # B6 moves it into the app too.
    system=ANALYSIS_SYSTEM_PROMPT,
    transcript=ANALYSIS_TRANSCRIPT_PROMPT,
    question="About the transcript above.\nQuestion: {question}\nAnswer with Yes or No only.",
    summary=SUMMARY_PROMPT,
)

PROMPTS: dict[str, Prompt] = {
    "plan": _PLAN,
    # Tuning-half finding: under "plan" the model reads "the speaker" as one
    # person and answers No to names and brands said by someone else (EN2002a's
    # "Mozilla", "Photoshop"; IS1009c's "Jana", "Francine" all near 0). The
    # transcript has no speaker turns, so the question has to say who counts.
    "anyone": replace(
        _PLAN,
        question=(
            "About the transcript above. It may have several speakers, and "
            '"the speaker" means any of them: answer Yes if it happens anywhere '
            "in the transcript.\nQuestion: {question}\nAnswer with Yes or No only."
        ),
    ),
}


@dataclass
class ExcerptResult:
    prefix_tokens: int
    prefill_seconds: float
    summary_seconds: float
    questions_seconds: float
    summary_text: str
    summary: dict[str, Any] | None
    p_yes: dict[str, float] = field(default_factory=dict)
    yes_no_mass: dict[str, float] = field(default_factory=dict)
    top_token: dict[str, str] = field(default_factory=dict)


class Analyzer:
    def __init__(self, prompt: Prompt, model: str = "8b") -> None:
        # Strict by default: mlx_lm.load passes no strict= and load_model's
        # default is True (confirmed by experiment in Phase 0).
        repo, revision = MODELS[model]
        loaded = load(repo, revision=revision)
        self.model, self.tokenizer = loaded[0], loaded[1]
        self.prompt = prompt
        self.yes_id, self.no_id = self._single_token("Yes"), self._single_token("No")
        self.eos = set(self.tokenizer.eos_token_ids)

    def _single_token(self, text: str) -> int:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        assert len(ids) == 1, f"{text!r} is {len(ids)} tokens"
        return ids[0]

    def _ids(
        self, messages: list[dict[str, str]], generation_prompt: bool
    ) -> list[int]:
        return list(
            self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=generation_prompt, enable_thinking=False
            )
        )

    def _prefix_messages(self, transcript: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.prompt.system},
            {
                "role": "user",
                "content": self.prompt.transcript.format(transcript=transcript),
            },
        ]

    def _suffix(self, prefix: list[int], messages: list[dict[str, str]]) -> list[int]:
        full = self._ids(messages, generation_prompt=True)
        # The canonical prefix check (PLAN.md, Phase 1): a template that rewrote
        # an earlier turn would leave the cached prefix describing other tokens.
        assert full[: len(prefix)] == prefix, "chat template rewrote the cached prefix"
        return full[len(prefix) :]

    def analyse(self, transcript: str, questions: dict[str, str]) -> ExcerptResult:
        messages = self._prefix_messages(transcript)
        prefix = self._ids(messages, generation_prompt=False)
        cache = make_prompt_cache(self.model)

        began = time.perf_counter()
        for i in range(0, len(prefix), PREFILL_STEP):
            self.model(mx.array(prefix[i : i + PREFILL_STEP])[None], cache=cache)
            mx.eval([c.state for c in cache])
        prefill_seconds = time.perf_counter() - began

        began = time.perf_counter()
        suffix = self._suffix(
            prefix, [*messages, {"role": "user", "content": self.prompt.summary}]
        )
        generated: list[int] = []
        for token, _ in generate_step(
            mx.array(suffix),
            self.model,
            max_tokens=SUMMARY_MAX_TOKENS,
            sampler=make_sampler(temp=0.0),
            prompt_cache=cache,
        ):
            # Typed mx.array, but generate_step yields y.item(), a Python int.
            token_id = int(token)
            if token_id in self.eos:
                break
            generated.append(token_id)
        summary_text = self.tokenizer.decode(generated)
        # offset, not a hand count: generate_step runs one step ahead.
        trim_prompt_cache(cache, cache[0].offset - len(prefix))
        summary_seconds = time.perf_counter() - began

        result = ExcerptResult(
            prefix_tokens=len(prefix),
            prefill_seconds=prefill_seconds,
            summary_seconds=summary_seconds,
            questions_seconds=0.0,
            summary_text=summary_text,
            summary=parse_summary(summary_text),
        )
        began = time.perf_counter()
        for key, question in questions.items():
            content = self.prompt.question.format(question=question)
            suffix = self._suffix(
                prefix, [*messages, {"role": "user", "content": content}]
            )
            logits = self.model(mx.array(suffix)[None], cache=cache)[0, -1].astype(
                mx.float32
            )
            trim_prompt_cache(cache, len(suffix))
            probs = mx.softmax(logits)
            pair = mx.softmax(logits[mx.array([self.yes_id, self.no_id])])
            result.p_yes[key] = float(pair[0])
            result.yes_no_mass[key] = float(probs[self.yes_id] + probs[self.no_id])
            result.top_token[key] = self.tokenizer.decode([int(mx.argmax(logits))])
        result.questions_seconds = time.perf_counter() - began
        assert cache[0].offset == len(prefix)
        return result


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_summary(text: str) -> dict[str, Any] | None:
    """The summary JSON with its keys and types checked, or ``None``.

    No retry here, unlike Phase 1's design: the evaluation reports how often
    the first attempt parses, which is what decides whether a retry earns its
    cost.
    """
    stripped = text.strip()
    if match := _FENCE.match(stripped):
        stripped = match.group(1)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("topic"), str) or not isinstance(
        data.get("summary"), str
    ):
        return None
    for key in ("topics", "action_items"):
        value = data.get(key)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            return None
    return {k: data[k] for k in ("topic", "summary", "topics", "action_items")}
