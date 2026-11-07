# Demo video plan

A 60–90 second video of the app for a social or portfolio post. The footage is captured by a Playwright script that drives the real app, so a retake is a re-run rather than a re-performance. The audio it transcribes is a scripted meeting synthesised with macOS `say`. This is a plan, not shipped code: nothing here changes `streamlit_app.py`, and the capture script lives in a scratchpad (see CLAUDE.md, Commands, Spikes) unless we decide to keep it.

## The message

One sentence carries the whole video: **a meeting recording goes in, and a timestamped transcript, a summary and a ranked question board come out, all on a Mac with nothing sent anywhere.** Every shot either shows that or gets cut.

What the video must not claim (each one is checked against CLAUDE.md before captions are written):

- **Offline from first use.** The first run downloads the speech model (~0.95 GB) and the analysis model (~3.9 GB). The accurate wording is "runs entirely on your Mac" or "nothing leaves the machine", not "no internet needed".
- **Speed without a source.** "~400× realtime" was measured on an M-series Mac (20 min in 2.55 s). Name the machine (M2 Pro) in the caption or leave the number out.
- **Verdicts.** The board ranks questions and does not answer them. Don't caption a high bar as "Yes" (see `BOARD_CAPTION`).
- **Languages.** English only. Output is lowercase and unpunctuated, and the video shows it as it is.
- **Real-time footage that has been sped up or cut.** Any cut that removes waiting time gets a visible "sped up" or "skipped" marker.

## Storyboard (~75 s)

| # | Time | Shot | On-screen caption |
|---|------|------|-------------------|
| 1 | 0–4 s | Title card over the app in its empty state, dark theme | "Meeting in. Transcript, summary and insights out. All on your Mac." |
| 2 | 4–12 s | Drop `team_meeting.wav` on the uploader. The player and the filename caption appear | "Upload audio or video, or record from the mic" |
| 3 | 12–20 s | Sidebar: turn on **Transcript analysis**. The questions expander appears under Transcribe and is opened briefly | "Ask your own yes/no questions" |
| 4 | 20–28 s | Click **Transcribe**. The progress bar runs, the transcript card lands, then the "Transcription complete" toast | "Speech → text with Granite Speech 5.0, on device" |
| 5 | 28–45 s | The analysis skeleton, then the Analysis card: zoom in on Topic, Summary and Action items | "Summary and action items by Granite 4.2 3B, also on device" |
| 6 | 45–58 s | Zoom in on the question board, scrolling it slowly | "Questions ranked by how likely the answer is yes" |
| 7 | 58–68 s | Edit one question in the box, click **Analyse**. Only the board refreshes | "Change a question and only the board re-runs" |
| 8 | 68–72 s | Hover over the two **Download** buttons | "Export as .txt and .md" |
| 9 | 72–78 s | End card: app name, repo URL, model names, "MLX · Apple Silicon" | — |

Shot 4 is the one that sells speed. A one-minute clip transcribes in under a second warm, so it needs no cut. Shot 5 includes the prefill and summary (several seconds). Cut it if it's longer than ~4 s on screen, and mark the cut.

## Demo audio

A ~60 s two-person meeting, scripted so that the default questions (`DEFAULT_QUESTIONS`) land in a deliberate spread rather than at random. The aim is a board that visibly ranks. Some questions should score high and some low, because a board where everything is near 0 or near 1 shows nothing.

| Question | Planted in the script? |
|----------|-------------------------|
| Asking a question | Yes: "Can we move the launch?" |
| Making a decision | Yes: "Let's go with the Thursday release." |
| Angry or frustrated | **No.** The tone stays calm, so one bar should be low |
| Company, product or brand | Yes: a vendor name |
| Money or a price | Yes: "about two thousand dollars" |
| A person by name | Yes: "Priya will own the rollout" |
| A date, time or deadline | Yes: "Thursday at ten" |
| A place | **No**, or only weakly |
| Commits to doing something | Yes: "I'll send the notes tonight" |

Steps:

1. Write the script in the scratchpad as alternating lines for two speakers, ~150 words.
2. Voice it with two distinct voices, e.g. `say -v "Daniel (English (UK))"` and `say -v "Eddy (English (US))"`, one AIFF per line. Concatenate them with short gaps and write a 16 kHz WAV with PyAV from the project venv. No `ffmpeg` CLI is needed, which matches the app's own decoding.
3. Run it through the app once before capture, and adjust the script until:
   - the summary reads naturally
   - the action items list both commitments
   - the board shows a clear spread
   - the transcript has no embarrassing mishearing in the first lines a viewer will read

   The model writes numbers as digits and expands contractions (CLAUDE.md, Languages, Surface quirks), so write the script with that in mind.
4. Keep the final WAV and the script text out of the repo unless we decide otherwise. `docs/` currently holds only the screenshot.

The README screenshot shows "a one-minute team meeting", so this can be the same story. If the source script for that screenshot still exists, start from it.

## Capture

Playwright (Python) runs in a scratchpad venv, never `uv add`, with Chrome as the browser, as in the A6 layout measurements.

- **App:** a fresh `uv run streamlit run streamlit_app.py` on a fixed port. **Warm it first:** one full transcribe and analyse in a throwaway browser context, so both models are in `@st.cache_resource` and the take shows warm timings, not a 4 GB download. Clear the toasts by reloading before the take.
- **Viewport:** 1470×900 with the sidebar open (input / transcript columns ~388 / 590 px, measured). That keeps the two-column grid readable on a phone after cropping. Dark theme via `color_scheme="dark"`.
- **Sharpness:** Playwright's built-in `record_video_dir` records at the viewport's CSS-pixel size and does not honour `device_scale_factor`, so text comes out soft. Before committing to a pipeline, do a 10 s test take both ways:
  - (a) `record_video_dir` as is
  - (b) a CDP `Page.startScreencast` at `device_scale_factor=2`, writing frames that are assembled afterwards

  Pick (b) only if (a) is visibly soft at the post's final size.
- **Pacing:** the script owns the timing. It waits on app state, not on sleeps: the card subheader, the toast text, the board caption. It then holds for a fixed beat (1–1.5 s) after each state so viewers can read. Use `locator.hover()` before clicks so the motion reads as a person's. Playwright draws no cursor, so either inject a small CSS cursor dot that follows `mousemove`, or accept no cursor and rely on zooms.
- **File input:** `set_input_files` on the uploader's `input[type=file]`. A drag-and-drop animation isn't worth faking.
- **Determinism:** greedy decoding in both models and a pinned revision of each. The same WAV should produce the same transcript, summary and board on every take, so a retake changes timing only. If a take differs in content, find out why before re-cutting.
- **Output:** one continuous take per run, plus a JSON log of timestamps for each state change (upload, toggle, click, card, toast, board). The edit uses the log to place captions and zooms without scrubbing.

## Edit and export

- **Tool:** any editor that does keyframed crop/zoom and captions (Screen Studio, DaVinci Resolve or iMovie). The timestamp log drives the cut list.
- **Formats:** 1920×1080 (16:9) for X, YouTube and portfolio sites, and a 1080×1350 (4:5) crop for LinkedIn and Instagram feeds. The 4:5 version leans on zooms (shots 5–6 especially), because the full grid at phone width is too small to read.
- **Sound:** Playwright records no audio. Options:
  - silent with burned-in captions, which suits feeds that autoplay muted
  - a quiet music bed
  - optionally, 2–3 s of the meeting audio under shot 2 so viewers hear what is being transcribed

  Captions are burned in either way.
- **Captions:** high contrast, at least 4.5:1 against what they sit on (the theme was measured to that line; the video should keep it). Every caption is held on screen long enough to read twice.
- **Length check:** if it runs over 90 s, cut shot 8 first, then shot 7.

## Steps

1. Write and voice the meeting script, then tune it against the app until the board shows a spread (Demo audio).
2. Do the sharpness test take, and choose `record_video_dir` or screencast (Capture).
3. Write the capture script with state-based waits and the timestamp log. Do three takes and keep the cleanest.
4. Write captions and check each one against "What the video must not claim".
5. Cut the 16:9 version, then the 4:5 crop.
6. Review on a phone, muted, before posting.

## Open questions

- Light or dark? The README screenshot is dark. Should a final shot flip the theme from the menu to show both?
- Should the capture script and meeting script be kept in the repo (e.g. under `docs/demo/`) so the video can be re-shot after a UI change, or thrown away after the cut? Keeping them means they need to survive `ruff` and the CI format check.
- Should the Record tab appear (a 2 s shot of the mic widget) even though the take uses an upload?
