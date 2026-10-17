# Plan: transcript analysis with Granite 4.2 8B

Status: **draft, not started.** Written 2026-10-17 for a later implementation session. Delete this file (or move it under `docs/`) once the work lands; CLAUDE.md is where the lasting record goes.

## Goal

After transcription, optionally run one local LLM over the transcript and show:

1. **Summary card**: a short topic label, a summary, key topics and action items. This is generated text, parsed from JSON.
2. **Yes/no question board**: user-editable questions, each answered as **P(yes)** read from the model's next-token probabilities, not from generated text. Run over the whole transcript, and later optionally per VAD segment, so a yes can be located in time.

Everything runs on-device; nothing leaves the machine. The design borrows from two references:

- **Deepgram Text Intelligence** (`/v1/read`) has four features: summarize, sentiment, topics, intents. Deepgram itself is **rejected**: it needs an API key and sends users' content to an outside service, and this app is offline by design.
- **IBM's "Speech to Decisions" Space** (`ibm-granite/speech-to-decisions`) runs Granite Switch 4.1 8B. Its yes/no board, its normalisation P(yes) = P(Yes) / (P(Yes) + P(No)), and its trick of computing the transcript's attention state once and reusing it for every question are the ideas taken here. Its live mic streaming, the Guardian safety check and the Switch adapters are **not** taken (see Out of scope).

## Model

| | |
|---|---|
| Repo | `ibm-granite/granite-4.2-8b-q4-mlx`, IBM's own MLX conversion, Apache 2.0 |
| Revision | `9047f073e46c527f70dade07bb55b1700d728e36` (the only revision as of 2026-09-02; pin it, for the same reproducibility reason as `MODEL_REVISION`) |
| Weights | 4.95 GB, one `model.safetensors`, 4-bit affine, group size 64 |
| Architecture | `model_type: granite`, `GraniteForCausalLM`. Dense decoder: 40 layers, 32 attention heads, 8 KV heads (GQA), hidden 4096, vocab 100,352, 131,072-token context. `mlx_lm.models.granite` exists in mlx-lm 0.32.2 |
| Chat format | ChatML (`<\|im_start\|>role\n...<\|im_end\|>`). This is **not** Granite Switch's `<\|start_of_role\|>` format, so don't copy the Space's string markers |

Why this one, for a 16 GB M2 Pro (decided in the planning session):

- `8b-q8` (9.34 GB) and `30b-q4` (16.47 GB) don't fit next to the app. The app's existing load is about 3–4 GB of OS, browser and Streamlit, about 1.5 GB of TurboCTC at a 30 s segment's peak, and decoded audio. Also, macOS lets the GPU use well under all 16 GB.
- `3b-q8` (3.89 GB) is the **fallback** if Phase 0 shows 8B is too slow or too tight. The constant makes it a one-line swap.
- Prefer it over the `lmstudio-community/granite-4.2-8b-MLX-4bit` build of the same model because it's org-maintained and has a single revision to pin.

### Facts verified in the planning session (re-check in Phase 0)

- **Thinking mode is on by default.** The template sets `enable_thinking` to True unless passed, and the generation prompt then ends `<|im_start|>assistant\n<think>\n`. With `enable_thinking=False` it ends `<|im_start|>assistant\n<think></think>`, so the answer's first token comes right after `</think>`. A `reasoning_effort` kwarg also exists (`"low"`/`"high"`); it's unused here.
- **Model card warning:** passing a plain string prompt to mlx-lm adds a spurious BOS token. Apply the chat template to get token ids and pass the **ids**, with no `add_special_tokens`.
- **Sampling:** `generation_config.json` says temperature 1.0 and top-p 0.95, but mlx-lm doesn't read it and defaults to greedy decoding. **Use greedy on purpose** (`make_sampler(temp=0.0)`), because a summary that changes between identical runs isn't useful.
- **Token ids** (from `tokenizer.json` at the pinned revision; each is a single token): `"Yes"` 9642, `"No"` 2822, `" Yes"` 7566, `" No"` 2360. After `<think></think>` there's no space, so the candidates are most likely `Yes`/`No`. **Phase 0 must confirm this from the model's real top-k next tokens, not assume it.**
- **mlx-lm 0.32.2 API:**
  - `mlx_lm.load(path_or_hf_repo, tokenizer_config=None, model_config=None, adapter_path=None, lazy=False, return_config=False, revision=None, trust_remote_code=False)`. It has `revision=` but **no `strict=`**; the underlying `mlx_lm.utils.load_model(..., strict: bool = True, ...)` defaults to strict.
  - `generate_step(prompt, model, *, max_tokens, sampler, logits_processors, max_kv_size, prompt_cache, prefill_step_size=2048, kv_bits, ...)`, plus `stream_generate`.
  - `mlx_lm.models.cache.make_prompt_cache(model, max_kv_size=None)`, `trim_prompt_cache(cache, num_tokens)`, `can_trim_prompt_cache(cache)`. `KVCache` is trimmable.
- **Dependencies:** mlx-lm 0.32.2 requires `mlx>=0.32.2` and `transformers>=5.7.0`, and the lock already has mlx 0.32.2 and transformers 5.17.0, so no resolution conflict is expected. It also brings back `sentencepiece`, `protobuf`, `pyyaml` and `jinja2`; CLAUDE.md records `mlx-lm` and `sentencepiece` leaving the lock, and that note needs revising.

## Phase 0: spike and measure (before writing app code)

Do this in a scratch script under the session scratchpad, not in the repo. The project's habit is measured figures, and several constants below depend on them.

1. `uv add mlx-lm` (it lands at 0.32.2), then load `LLM_ID` at `LLM_REVISION` via `mlx_lm.load`. Record: cold download time, load time, resident memory (`mx.get_active_memory()`, `mx.get_peak_memory()`), and that loading is strict (a key-mismatch should raise).
2. Build a long transcript: tile real app output to about 30 minutes, roughly 5–6K words of lowercase unpunctuated text. Measure **prefill speed** (tokens/s) at 1K, 4K, 8K and 16K tokens, **generation speed** (tokens/s), and **peak memory** at each. Estimated KV cache is 40 layers × 2 (keys and values) × 8 heads × 128 dims × 2 bytes ≈ **160 KB per token**: about 1.3 GB at 8K tokens and about 2.6 GB at 16K. Confirm by measurement.
3. Measure the **whole process at its peak** with the speech model also resident, as the real app has it: TurboCTC transcription, then the LLM. Decide `MAX_ANALYSIS_TOKENS` from this. With VAD on, the app accepts any length of audio, and a 3-hour clip is about 40K tokens (about 6.5 GB of KV), so analysis needs its own limit. If it's too tight, try `kv_bits=8` in `generate_step` and measure the quality and memory difference before taking it.
4. Run the yes/no readout on 3–4 hand-labelled transcripts. Confirm that `Yes`/`No` (or the spaced forms) are the top next tokens after `<think></think>`, and sum their probability mass. If the mass is low, the model is trying to say something else, and the prompt needs changing before the number means anything.
5. Check the **cache equivalence**: logits from reusing the cached transcript plus a question suffix must match logits from the full prompt run from scratch, within bf16 tolerance. Also check that `trim_prompt_cache` restores the cache exactly between questions: question B's logits after question A then a trim must equal B's logits on a fresh copy of the cache.
6. Time 10 questions on a 30-minute transcript. Time the per-segment variant too: about 60 segments × 10 questions. This decides whether Phase 3 is worth doing.

**Decision point after Phase 0:**
- 8B vs 3B.
- `MAX_ANALYSIS_TOKENS`.
- Whether Phase 3 ships.

Write the measured numbers into CLAUDE.md in the same style as the TurboCTC memory series.

## Phase 1: summary card

### Code (`streamlit_app.py`)

- **Constants:** `LLM_ID`, `LLM_REVISION` (full 40-hex hash), `MAX_ANALYSIS_TOKENS` (from Phase 0), `ANALYSIS_MAX_NEW_TOKENS` (e.g. 512).
- **`load_llm(model_id, revision)`:** `@st.cache_resource`, wrapping `mlx_lm.load(model_id, revision=revision)`, loaded lazily like the other models. Its docstring should say why there's no `strict=` (the default is strict, and a contract test pins that). Return `(model, tokenizer)`.
- **`AnalysisResult(TypedDict)`:** `topic: str`, `summary: str`, `topics: list[str]`, `action_items: list[str]`, plus `questions: list[tuple[str, float]]` from Phase 2.
  - Deepgram's separate "intents" are dropped: for this app's audio (meetings, calls, memos), action items are the useful form, and the yes/no board covers the rest.
  - **No confidence numbers on generated fields.** A number the LLM writes is made up, not measured. Only Phase 2's probabilities are real.
- **`_analysis_prompt_ids(tokenizer, transcript) -> list[int]`:**
  - A system turn holding the instructions.
  - A user turn holding the transcript, delimited, and stating that it is **lowercase, unpunctuated ASR output that may contain misheard words** (the same caveat IBM's Space gives).
  - Then the generation prompt with `enable_thinking=False`, applied with `tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` to get ids.
  - Feed the transcript **without** the `[M:SS - M:SS]` prefixes: they cost tokens and invite made-up timestamps. Strip them from `PipelineResult["transcript"]`, or better, have `run_pipeline` also return the plain segment texts (see Data below).
  - The transcript is data, not instructions: keep it in the user turn behind a delimiter, and the instructions in the system turn. Prompt injection from audio is low-stakes for a local app, but the separation costs nothing.
- **`summarize_transcript(model, tokenizer, transcript) -> dict`:**
  - Greedy `generate_step`/`stream_generate`, `max_tokens=ANALYSIS_MAX_NEW_TOKENS`.
  - Parse JSON: strip a ```json code fence if present, `json.loads`, validate the keys and types.
  - **One retry** on a parse or validation failure, appending the bad output plus a "return only valid JSON" turn. mlx-lm has no schema-constrained decoding as far as the planning session found; check `logits_processors` before building one.
  - If the retry also fails, raise `RuntimeError` with a readable one-liner, following the project's `RuntimeError` rule.
- **Limit check:** count the prompt's tokens first. Over `MAX_ANALYSIS_TOKENS`, raise `RuntimeError(f"Transcript is too long to analyse ({n} tokens; the limit is {MAX_ANALYSIS_TOKENS}).")`. Don't truncate silently: a summary of the first hour labelled as the whole thing is worse than none. Chunked map-reduce summaries are out of scope for v1.

### Data

- Extend `PipelineResult` with `segments: list[Segment]`, where `Segment` is `TypedDict(start: float, end: float, text: str)`, alongside the existing `transcript`.
  - Phase 1 joins the `text`s.
  - Phase 3 needs the segments individually.
  - The download and card keep using `transcript` unchanged.
- **This changes `TestRunPipeline`'s exact-key-set assertion** (`{"transcript"}`). Update it on purpose, to `{"transcript", "segments"}`, rather than loosening it.

### UI

- **Sidebar:** a second `_labeled_toggle("Transcript analysis", help=..., key="use_analysis", value=False)`. It's **off by default**, so the default app, its memory and every existing test are unchanged. Help text: "Summarises the transcript and answers yes/no questions with a local Granite 4.2 8B model (~5 GB download on first use)."
  - The sidebar model line gains a second `:small[]` line for `LLM_ID`, shown only when analysis is on. Keep `:small[]` rather than `st.caption`; see CLAUDE.md's contrast note.
  - `test_sidebar_holds_settings_and_app_info_only` counts markdown elements and will need updating.
- **Run order inside the existing `try`:**
  1. Decode.
  2. Speech model, then VAD.
  3. `run_pipeline`.
  4. **Store `result`/`result_stem` first.**
  5. Then, if analysis is on: spinner "Loading language model..." (`show_time=True`; a cold 4.95 GB fetch takes minutes), then "Analysing transcript...".

  Because the transcript is committed before analysis starts, **an analysis failure must not lose the transcript.** Catch analysis's `RuntimeError` separately and show it inside the analysis card's slot, not as the run's error. Pin this with an AppTest.
- **Card:** render a second bordered card **below** the transcript card in `transcript_slot`, titled "Analysis":
  - Topic as a bold line.
  - Summary as `st.markdown` text.
  - Topics as a small inline list.
  - Action items as a bulleted list.
  - No `st.success`/`st.warning` banner, following the transcript card's rule.
  - Keep the slot's shape constant (the `feedback` placeholder at index 0; see CLAUDE.md's Run feedback note about delta paths and remounts).
- **Invalidation:**
  - `result` stays keyed by `_last_input_key` as today.
  - `analysis` gets its own key: `_analysis_key = (_last_input_key, questions_tuple)`.
  - Toggling analysis on, or editing questions, **after** a transcript exists must not re-transcribe. Show a secondary "Analyse" button (`type="secondary"`) in the analysis slot that runs analysis only over `st.session_state.result`.
  - Pop `analysis` whenever `_last_input_key` changes, in the same loop that pops `result`/`result_stem`.
- **Download:** v1 keeps `<stem>_transcription.txt` unchanged. An optional second button gives `<stem>_analysis.md`, but keep that out of v1 unless trivial.

## Phase 2: yes/no question board (whole transcript)

### Code

- **`DEFAULT_QUESTIONS`:** adapted from the Space's defaults, minus its joke "cats" question:
  - "Is the speaker asking a question?"
  - "Is the speaker making a decision?"
  - "Is the speaker angry or frustrated?"
  - "Does the speaker name a company, product or brand?"
  - "Does the speaker mention money or a price?"
  - "Does the speaker mention a person by name?"
  - "Does the speaker mention a date, a time or a deadline?"
  - "Does the speaker mention a place?"
  - "Does the speaker commit to doing something?"
- **`MAX_QUESTIONS`:** e.g. 20. The cost is a short forward pass per question, so it's a UI limit more than a speed one; set it from Phase 0's timing.
- **`_yes_no_ids(tokenizer) -> tuple[int, int]`:** resolve the answer tokens once at load and **assert each encodes to exactly one token.** Use the spelling Phase 0 confirmed (`"Yes"`/`"No"` expected).
- **`answer_questions(model, tokenizer, transcript, questions) -> list[tuple[str, float]]`:**
  1. Build the prompt for question 0 and find the shared prefix: the token ids up to and including the transcript's user turn. Assert that every question's prompt starts with exactly that prefix, because chat templates can rewrite earlier turns (e.g. the `<think></think>` handling of assistant turns). If one doesn't, fall back to a full prompt per question and warn (`RuntimeWarning`, matching the VAD fallbacks' habit of never downgrading silently).
  2. `cache = make_prompt_cache(model)`, then compute the prefix's attention state in `prefill_step_size` chunks: `model(chunk[None], cache=cache)`, then `mx.eval` over the cache state.
  3. For each question:
     - Run its suffix (question turn plus `<|im_start|>assistant\n<think></think>`) once: `logits = model(suffix[None], cache=cache)[0, -1]`.
     - Compute `p_yes = softmax(logits[[yes_id, no_id]])[0]`.
     - Call `trim_prompt_cache(cache, len(suffix))` to restore the cache.
     - Assert `can_trim_prompt_cache(cache)` once up front.
  4. Return the questions in **the user's order**. The UI sorts.
- **Prompt for each question:** a user turn of `f"Question: {q}\nAnswer with Yes or No only."` about "the transcript above".
  - Phase 0 may show a system turn or a different wording raises the Yes+No mass; tune there, not here.
  - Optional diagnostic: also compute and log the combined Yes+No mass and log a warning when it's below a threshold, say 0.5.

### UI

- **Questions input:** a `st.text_area("Questions", value="\n".join(DEFAULT_QUESTIONS), help="One yes/no question per line.")` in the **input column**, inside an `st.expander("Analysis questions")` under the Transcribe button, shown only when analysis is on. It goes there because the run depends on it, and CLAUDE.md's layout rule keeps run inputs in main, not the sidebar.
  - Strip blank lines and drop duplicates (like the Space's `clean_nouls`).
  - Cap at `MAX_QUESTIONS` with an inline caption if the user goes over.
- **Board,** inside the Analysis card under the summary:
  - One row per question, **sorted by P(yes), highest first**, each a label plus a bar.
  - Prefer `st.dataframe(..., column_config={"P(yes)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")}, hide_index=True)`, which is one element and easy to test. Check it renders at the 720 px card width before settling on it over per-row `st.progress`.
  - **No 0.5 "yes" verdict and no colour flip.** The base instruct model isn't the Space's trained `decision` adapter, so its probabilities are less calibrated; present them as a ranking. Say so in one caption under the board.
- **Progress:** "Answering question i of n" through the same `feedback` progress bar, following `on_progress`'s before-each-unit contract.

## Phase 3 (optional, gated on Phase 0 timing): per-segment answers

- For each VAD segment, compute that segment's text into a fresh cache, then ask the questions as in Phase 2.
- Result: a segment × question matrix of P(yes).
- UI: for a chosen question (a `st.selectbox` in the card), list the segments whose P(yes) is highest, as `[M:SS - M:SS] 0.91 text…`, linked to the timestamps the transcript already shows.
- Also opens per-segment sentiment for free, e.g. "Is the speaker angry or frustrated?" per segment.
- Cost is segments × questions short forward passes. Each is bound by memory bandwidth (reading roughly 5 GB of weights per pass), so it may be about 25 ms per pass on an M2 Pro. **That figure is an estimate, not a measurement.** If Phase 0 shows it's too slow, batch the questions along the batch axis over a copied cache, or skip the phase.
- VAD off yields one segment, so Phase 3 collapses to Phase 2. Hide the selector then.

## Tests

Follow the existing conventions (CLAUDE.md, Tests section):

- **`test_streamlit_app.py`:** pure helpers are parametrized functions, stateful code goes in `Test*` classes, and `st` is mocked wholesale.
- **`test_app_ui.py`:** AppTest patches the **shared upstream import**, not `streamlit_app.*`.

### Unit

- `test_llm_revision_is_pinned`: a full 40-hex hash, like `test_model_revision_is_pinned`.
- JSON parsing: valid JSON; JSON inside a code fence; JSON with stray prose around it; missing keys and wrong types (raise); one retry then `RuntimeError`.
- Yes/no normalisation from fake logits: equal logits give 0.5; a large Yes margin gives ≈ 1. Only the two ids matter, so other vocab entries must not change the result.
- Prefix-mismatch fallback: a fake tokenizer whose template changes the prefix gives the full-prompt path plus a `RuntimeWarning`.
- `MAX_ANALYSIS_TOKENS` refusal gives a `RuntimeError` with the token count in the message.
- `TestRunPipeline`: the new `segments` key, and an exact key-set update.
- `TestLoadLlm` decorated with `@patch("streamlit_app.st", new=MagicMock(spec=[]))`, the same guard `TestLoadModel` uses.

### Contract (real installed `mlx_lm`, no download)

- `mlx_lm.load` has a named `revision` parameter.
- `mlx_lm.utils.load_model`'s `strict` parameter defaults to `True`. That's the only thing standing between a renamed key and randomly initialised layers, since `load()` doesn't expose it.
- `mlx_lm.models.granite` imports, and `make_prompt_cache`/`trim_prompt_cache`/`can_trim_prompt_cache` exist.
- **Cache-equivalence test with a tiny random Granite model,** like `TestMlxVadInternals`:
  - Build `mlx_lm.models.granite.Model(ModelArgs(...))` with 2 layers and a small hidden size: no weights to download, but the real classes.
  - Assert (a) cached prefix plus suffix logits ≈ full-prompt logits, and (b) after trimming question A, question B's logits match a fresh cache.
  - Drop the trim and (b) fails. That's the point: it's the only CI coverage of the cache logic.

### Real weights

`test_analyses_a_transcript_with_real_weights`:

- Gated on `huggingface_hub.try_to_load_from_cache(LLM_ID, "model.safetensors", revision=LLM_REVISION)`, exactly as the TurboCTC smoke test is gated, so CI never fetches 4.95 GB.
- Assert only structure: valid keys, a non-empty summary, probabilities in [0, 1]. Plus one strong semantic signal, e.g. a transcript that clearly says "the meeting is on friday at three" gets P(yes) > 0.5 on the date/time question.
- Don't pin exact wording, for the same reason the TurboCTC test doesn't.

### AppTest

- Patch `mlx_lm.load` to return a spec'd fake `(model, tokenizer)`.
- Analysis off (default): no LLM load and no Analysis card. The existing tests pass unchanged.
- Analysis on: one Analysis card in the transcript column below the transcript card, the board present, and `load` called once with `revision=LLM_REVISION`.
- An analysis failure keeps the transcript card plus an error inside the analysis area, and `result` stays in session state.
- Toggling analysis on after a run shows "Analyse" and does **not** re-transcribe: the speech model's `generate` call count is unchanged.
- Editing questions after an analysis invalidates `analysis` but not `result`.

## Docs and housekeeping

- **CLAUDE.md:**
  - Dependencies: `mlx-lm` is back, and why, with the version floor once chosen. Revise the "`mlx-lm` is no longer imported anywhere" passages, including the note that `_supports_mlx_vad` is the only probe over library internals, if Phase 2 relies on cache internals.
  - Models: the new entry.
  - Architecture: the new functions.
  - UI Layout: the analysis toggle, questions expander and Analysis card.
  - Languages: Phase 0's memory and speed figures and `MAX_ANALYSIS_TOKENS`.
  - Tests.
  - The Project Overview currently says "Transcription only", which becomes untrue.
- **README.md:** the feature, its memory cost and the first-use download.
- **Markdown fences:** `ruff format --check .` formats ```python fences in `.md` files (ruff ≥ 0.16.0, per CLAUDE.md), and **this file is checked too while it sits at the repo root.** That's why it has no Python fences. Keep it that way, or format any you add.
- **Commits** (conventional, with scopes; see CLAUDE.md Git):
  - `chore(deps): add mlx-lm for transcript analysis`
  - `feat(pipeline): summarise transcripts with granite 4.2 8b`
  - `feat(pipeline): answer yes/no questions from cached transcript prefill`
  - `feat(ui): add transcript analysis card and questions`
  - `docs: document transcript analysis`
  - Commit when asked; the user pushes (no `git push`).
- **Releases:** don't bump `version` as part of this; the bump is the release action and is the user's call.

## Out of scope (decided in the planning session)

- **Deepgram API:** a cloud service and a key, which breaks the app's offline design.
- **Granite Switch** (`granite-switch-4.1-{8b,30b}`): no MLX build exists, and its Transformers/vLLM checkpoint won't fit in 16 GB anyway. Its `decision` adapter is what the plain instruct model only approximates here.
- **Guardian safety check:** Granite 4.2 has no Guardian adapter, and a separate `granite-guardian-4.1-8b` (about 5 GB at 4-bit) won't fit next to the 8B model. Questions like "Does the speaker use profanity?" on the board are the cheap approximation.
- **Live mic streaming** (the Space's re-transcribe-every-0.5 s loop): a different product, and a poor fit for Streamlit's rerun model.
- **Restoring casing and punctuation:** a plausible later use of the same LLM, but a separate feature.
- **Chunked summaries for transcripts over the token limit.**

## Risks and open questions

- **Calibration:** P(yes) from a base instruct model is a ranking, not a probability you can rely on. Phase 0 step 4 measures how much it's off; the UI must not present it as a verdict.
- **ASR errors carry through:** e.g. the fixture's `shapeare` and `libvox`. The prompt warns the model; the summary may still repeat misheard words.
- **Memory at the edge of 16 GB:** both models stay cached after a run. If Phase 0 shows pressure, consider clearing the speech model's cached resource before analysis. But that costs a reload on the next Transcribe; measure before choosing.
- **The model loads on first use:** the first analysis on a cold cache is a 4.95 GB download. The toggle's help text and the `show_time` spinner must make that visible.
- **Template drift:** a future revision of the chat template could change the `<think></think>` suffix or the prefix layout. The prefix-equality assertion and the Yes+No mass warning are the tripwires; the revision pin is what keeps them quiet.
