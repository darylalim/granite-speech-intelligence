# Demo video plan

A 60–90 second video of the app for a social or portfolio post. The footage is captured by a Playwright script that drives the real app, so a retake is a re-run rather than a re-performance. The audio it transcribes is a scripted meeting synthesised with macOS `say`. This is a plan, not shipped code: nothing here changes `streamlit_app.py`, and the capture script lives in a scratchpad (see CLAUDE.md, Commands, Spikes) unless we decide to keep it.

## The message

One sentence carries the whole video: **a meeting recording goes in, and a timestamped transcript, a summary and a ranked question board come out, all on a Mac with nothing sent anywhere.** Every shot either shows that or gets cut.

What the video must not claim (each one is checked against CLAUDE.md before captions are written):

- **Offline from first use.** The first run downloads the speech model (~0.95 GB) and the analysis model (~3.9 GB). The accurate wording is "runs entirely on your Mac" or "nothing leaves the machine", not "no internet needed".
- **Speed without a source.** CLAUDE.md's "~400× realtime" (20 min in 2.55 s) was measured on an M2 Pro. The demo is recorded on an **M2 Max (32 GB)**, where the meeting below transcribes in 0.56 s and analyses in 3.0 s. A speed caption names the machine it was measured on, or it leaves the number out.
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
| 7 | 58–68 s | In the questions box, replace "Does the speaker mention a place?" with "Is a bug or problem discussed?" and click **Analyse**. Only the board refreshes, and the new question goes to the top | "Ask what matters to you. Only the board re-runs" |
| 8 | 68–72 s | Hover over the two **Download** buttons | "Export as .txt and .md" |
| 9 | 72–78 s | End card: app name, repo URL, model names, "MLX · Apple Silicon" | — |

Shot 4 is the one that sells speed. The meeting transcribes in 0.56 s warm, so it needs no cut. Shot 5 includes the prefill, summary and board (3.0 s measured), so it fits uncut as well.

Shot 7 does two jobs. It shows that the questions can be edited, and it removes the board's one misleading score: "place" scores 0.85 on a meeting that mentions no place (see Demo audio). Swapping it for a question the meeting is actually about is the honest fix, and it's what the feature is for.

## Demo audio

**Done (2026-11-07).** A two-person meeting, tuned against the app over two script revisions and a brand-name A/B. The script is the source of truth: the WAV is rebuilt from it in seconds, so neither the audio nor the voiced lines are committed.

### Script

| Speaker | Line |
|---------|------|
| A | Morning Tom. Quick question for you. Can we move the launch of the new dashboard to next week? |
| B | We could, but the payment screen still fails on older phones. What would we gain? |
| A | Time to fix that bug properly. The Brightpay team also wants a final review before we ship. |
| B | Fair enough. Then let us go with the Thursday release, at ten in the morning. |
| A | Agreed. One more question. What about the budget for extra testing? |
| B | The test devices cost about two thousand dollars. I think that is reasonable. |
| A | Good. Maya will own the rollout, and she will check the release notes with Brightpay on Wednesday. |
| B | I will send the meeting notes tonight, and I will book the test devices tomorrow. |
| A | Perfect. Is there anything else before we finish? |
| B | No, that is everything. Thanks, see you on Thursday. |

### How to build it

- A is voiced by `say -v "Daniel (English (UK))"` and B by `say -v "Eddy (English (US))"`, one AIFF per line.
- Each line is decoded with the app's own `_decode_audio`, then joined with 0.5 s of leading silence and **1.2 s** between lines.
- The result is written as a 16-bit, 16 kHz mono WAV: **62.6 s**.
- No `ffmpeg` CLI is needed, which matches the app's own decoding.

### What the app produces (VAD on, M2 Max)

- **Transcript (0.56 s):** ten segments, one per line, `[0:00 - 0:06]` to `[0:56 - 1:01]`, word for word except that the first "Brightpay" comes out as "bridepay".
- **Analysis (3.0 s, summary plus board):**
  - Topic: "dashboard launch and testing".
  - Three action items: Maya's review with Brightpay, sending the notes, booking the devices.
- **Board on the defaults:**

| P(yes) | Question | Present in the meeting? |
|--------|----------|--------------------------|
| 1.00 | Date, time or deadline | Yes |
| 1.00 | Money or a price | Yes |
| 0.99 | Commits to doing something | Yes |
| 0.99 | Person by name | Yes |
| 0.97 | Company, product or brand | Yes |
| 0.93 | Making a decision | Yes |
| **0.85** | **Place** | **No.** A false positive, which shot 7 replaces |
| 0.82 | Asking a question | Yes |
| 0.00 | Angry or frustrated | No |

The shot-7 replacement, "Is a bug or problem discussed?", scores **1.00**.

### What tuning taught

- **Questions have to be marked with words.** TurboCTC writes no punctuation, so "can we move the launch" reads as a plain clause. With three spoken questions the first draft scored **0.20** on "asking a question". Adding "Quick question for you" and "One more question" raised it to **0.82**.
- **Pauses set the transcript's layout.** VAD pads each span by 0.3 s on both sides and merges spans whose padded gap is under 0.5 s, so a raw pause under ~1.1 s disappears. At 0.6 s gaps the whole meeting merged into one span, which the 30 s cap then split into two ~27 s blocks at an arbitrary point. At 1.2 s, each turn is its own timestamped line.
- **"Place" is not the brand name.** The first draft's "Northwind", once transcribed as "north wind", was the obvious suspect, but removing it left "place" at 0.85. The evaluation found the same thing: these scores rank questions, they don't decide them.
- **Made-up brand names are what TurboCTC mishears.** With the `say` voices:
  - Brightpay: 1 of 2 mentions exact, brand score 0.97
  - Paywell: transcribed as "paywall", 0.62
  - Copperline: "copperine", 0.68
  - Zenpay: "zenpe", 0.87

  Brightpay is kept. "bridepay" still reads as a brand, and it appears in line 3, not in the opening lines a viewer reads first.
- **Write the script the way the model will output it:** contractions spelled out ("let us", "I will"), because the model expands them anyway, and numbers spoken in words, because it writes them as digits ("10", "$2000").

## Capture

Playwright (Python) runs in a scratchpad venv, never `uv add`, with Chrome as the browser, as in the A6 layout measurements.

- **App:** a fresh `uv run streamlit run streamlit_app.py` on a fixed port. **Warm it first:** one full transcribe and analyse in a throwaway browser context, so both models are in `@st.cache_resource` and the take shows warm timings, not a 4 GB download. Clear the toasts by reloading before the take.
- **Viewport:** 1470×900 with the sidebar open (input / transcript columns ~388 / 590 px, measured). That keeps the two-column grid readable on a phone after cropping. Dark theme via `color_scheme="dark"`.
- **Sharpness: frame-grab at DSF 2.** Decided 2026-11-07 by test takes of the transcript card landing, compared with the card zoomed to fill a 1080-wide 4:5 frame (×1.94 from CSS px):
  - **(a) `record_video_dir`: rejected.** It records at CSS-pixel size (1470×900) whatever the `device_scale_factor`. Upscaled ×1.94 the text is visibly soft, and its low-bitrate VP8 (0.4 MB for ~8 s) adds ringing around the letters. It also needs Playwright's own `ffmpeg` download (`playwright install ffmpeg`).
  - **(b) CDP `Page.startScreencast`: rejected.** Under DSF emulation it never sends a frame larger than the CSS viewport, headless or headed, even with `maxWidth`/`maxHeight` set to device size. A lossless 1470×900 frame upscaled ×1.94 is as soft as (a) without the ringing, so the problem is resolution, not compression.
  - **(c) CDP `Page.captureScreenshot` in a timed loop: chosen.** It needs `clip: {x: 0, y: 0, width, height, scale: 2}`; without the clip it, too, returns CSS-size frames. It returns true 2940×1800 frames at a steady **20 fps** (50 ms each, i.e. three display refreshes; median gap 50 ms, max 78 ms, no failed captures). The frames are resampled to constant 30 fps and encoded with PyAV's bundled `libx264` (`crf 14`, `preset slow`, `tune stillimage`): 4.8 s came to 0.3 MB, and at the 4:5 zoom it can't be told apart from a lossless DSF 2 PNG. CSS size runs at 30–45 fps, which is the only reason to trade the resolution back.

  Mechanics that matter:
  - The capture loop and the shot run together on Playwright's **async** API; the sync API can't capture while it clicks.
  - Start capturing only after the app has loaded. Capturing during `page.goto` fails with "Not attached to an active page".
  - 20 fps is enough for a UI that mostly holds still. The zooms and pans are keyframed in the editor from the 2940×1800 master, so they move at the export's frame rate, not the capture's. The toast's slide-in and the progress bar are the only motion recorded at 20 fps.
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

1. ~~Write and voice the meeting script, then tune it against the app until the board shows a spread (Demo audio).~~ **Done.**
2. ~~Do the sharpness test take, and choose `record_video_dir` or screencast (Capture).~~ **Done:** neither; frame-grab at DSF 2.
3. **Next:** write the capture script with state-based waits and the timestamp log. Do three takes and keep the cleanest.
4. Write captions and check each one against "What the video must not claim".
5. Cut the 16:9 version, then the 4:5 crop.
6. Review on a phone, muted, before posting.

## Open questions

- Light or dark? The README screenshot is dark. Should a final shot flip the theme from the menu to show both?
- Should the capture script and the WAV-building script be kept in the repo (e.g. under `docs/demo/`) so the video can be re-shot after a UI change, or thrown away after the cut? The meeting script itself now lives in this file. Keeping the code means it has to pass `ruff` and the CI format check, and the tuning scripts import `streamlit_app`, so they would need the repo root on `PYTHONPATH` (pytest's `pythonpath` setting doesn't apply to them).
- Should the Record tab appear (a 2 s shot of the mic widget) even though the take uses an upload?
