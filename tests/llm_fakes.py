"""Fakes for the analysis LLM, shared by the unit tests and the AppTests.

The LLM is faked at three seams: a tokenizer whose ids are characters, a KV
cache that is just an offset, and a generate_step that replays a script. The
real classes are exercised by TestGraniteLlmContract on a tiny random model.
"""

from collections.abc import Iterator
from typing import Any, ClassVar

import mlx.core as mx

# FakeTokenizer's control tokens; every character token is its code point, so
# all of these sit above anything ASCII text produces.
ROLE_IDS = {"system": 1001, "user": 1002, "assistant": 1003}
END_ID, GEN_ID, EOS_ID = 1004, 1005, 0
YES_ID, NO_ID = 1010, 1011


class FakeTokenizer:
    """Characters as token ids: decode is chr, and a prompt's prefix stays put.

    The real template's only property the app relies on is that adding turns
    leaves the earlier turns' ids unchanged; this one has it by construction.
    """

    eos_token_ids: ClassVar[set[int]] = {EOS_ID}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        assert enable_thinking is False
        ids: list[int] = []
        for message in messages:
            ids += [ROLE_IDS[message["role"]], *map(ord, message["content"]), END_ID]
        return [*ids, GEN_ID] if add_generation_prompt else ids

    def decode(self, ids: list[int]) -> str:
        return "".join(map(chr, ids))

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return {"Yes": [YES_ID], "No": [NO_ID]}.get(text, tokens(text))


class RewritingTokenizer(FakeTokenizer):
    """A template that rewrites the start once a task turn follows the prefix."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        ids = super().apply_chat_template(
            messages, add_generation_prompt, enable_thinking
        )
        return [END_ID, *ids] if len(messages) > 2 else ids


class FakeKVCache:
    """Just the KVCache surface the app and trim_prompt_cache touch."""

    def __init__(self) -> None:
        self.offset = 0
        self.state = mx.array(0)

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n


def fake_llm(ids: mx.array, cache: list[FakeKVCache]) -> mx.array:
    for c in cache:
        c.offset += ids.shape[1]
    return mx.zeros((1, ids.shape[1], 8))


def tokens(text: str) -> list[int]:
    return [ord(c) for c in text]


VALID_JSON = '{"topic": "t", "summary": "s", "topics": ["a"], "action_items": []}'
VALID_SUMMARY = {"topic": "t", "summary": "s", "topics": ["a"], "action_items": []}


class ScriptedGenerate:
    """Stands in for generate_step, replaying one scripted reply per call.

    Like the real one it advances the cache by the prompt, then computes one
    step ahead of each token it yields — the extra token the trim must remove
    by reading `offset` rather than counting what the loop consumed.
    """

    def __init__(self, *replies: list[int]) -> None:
        self.replies = list(replies)
        self.prompts: list[list[int]] = []

    def __call__(
        self,
        prompt: mx.array,
        model: Any,
        *,
        max_tokens: int,
        sampler: Any,
        prompt_cache: list[FakeKVCache],
    ) -> Iterator[tuple[int, None]]:
        self.prompts.append([int(t) for t in prompt])
        reply = self.replies[len(self.prompts) - 1]
        for c in prompt_cache:
            c.offset += prompt.size
        for token in reply[:max_tokens]:
            for c in prompt_cache:
                c.offset += 1
            yield token, None

    def prompt_text(self, i: int) -> str:
        return "".join(chr(t) for t in self.prompts[i] if t < 1000)


class FakeLlm:
    """A whole fake model for the AppTests, which cannot patch streamlit_app.

    It builds its own cache (mlx_lm's make_prompt_cache defers to
    `make_cache`), and scores a question by a fixed Yes margin looked up from
    the question's text, so the board's sort order is known in advance.
    """

    def __init__(self, margins: dict[str, float] | None = None) -> None:
        self.margins = margins or {}

    def make_cache(self) -> list[FakeKVCache]:
        return [FakeKVCache()]

    def __call__(self, ids: mx.array, cache: list[FakeKVCache]) -> mx.array:
        fake_llm(ids, cache)
        text = "".join(chr(int(t)) for t in ids[0] if int(t) < 1000)
        margin = next(
            (m for q, m in self.margins.items() if f"Question: {q}\n" in text), 0.0
        )
        logits = mx.full((1, ids.shape[1], 1100), -30.0)
        logits[0, -1, YES_ID] = margin
        logits[0, -1, NO_ID] = 0.0
        return logits
