# Demo video plan

A 60–90 second video of the app for a social or portfolio post. The footage is captured by a Playwright script that drives the real app, so a retake is a re-run rather than a re-performance. The audio it transcribes is a scripted meeting synthesised with macOS `say`. Nothing here changes `streamlit_app.py`. The scripts that rebuild the video are in `docs/demo/` so it can be re-shot after a UI change; see Reproduce.

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
| 3 | 12–20 s | Click **Transcribe** with analysis still off. The transcript card lands, ten timestamped lines, then the "Transcription complete" toast | "Speech → text with Granite Speech 5.0, on device" |
| 4 | 20–28 s | Sidebar: turn on **Transcript analysis**. The questions expander appears and is opened. Because a transcript already exists, the app offers **Analyse** instead of transcribing again; click it | "Add analysis and ask your own yes/no questions" |
| 5 | 28–45 s | The analysis skeleton, then the Analysis card: zoom in on Topic, Summary and Action items | "Summary and action items by Granite 4.2 3B, also on device" |
| 6 | 45–58 s | Zoom in on the question board, scrolling it slowly | "Questions ranked by how likely the answer is yes" |
| 7 | 58–68 s | In the questions box, replace "Does the speaker mention a place?" with "Is a bug or problem discussed?" and click **Analyse**. Only the board refreshes, and the new question goes to the top | "Ask what matters to you. Only the board re-runs" |
| 8 | 68–72 s | Hover over the two **Download** buttons | "Export as .txt and .md" |
| 9 | 72–78 s | End card: app name, repo URL, model names, "MLX · Apple Silicon" | — |

Shot 3 is the one that sells speed. The meeting transcribes in 0.56 s warm, so it needs no cut. **Transcribe comes before analysis is turned on**, which reverses the first draft. With analysis on from the start, the app draws the analysis skeleton (~1,100 px) *above* the transcript card, by design, so the card lands below the fold and the transcription moment is never on screen (seen in the first full take). Turning analysis on afterwards is a real path through the app, and it shows one more feature: analysis never re-transcribes. Shot 5 includes the prefill, summary and board (3.0 s measured), so it fits uncut as well.

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

Playwright (Python) runs in an ephemeral `uv run --no-project --with` environment, never `uv add`, with the installed Chrome as the browser, as in the A6 layout measurements.

- **App:** a fresh `uv run streamlit run streamlit_app.py --server.port 8599 --server.headless true --client.toolbarMode minimal`. The last flag removes the dev-mode "Deploy" button from the corner of every frame; it is a launch flag, so `.streamlit/config.toml` is unchanged. **Warm it first:** one full transcribe and analyse in a throwaway browser context, so both models are in `@st.cache_resource` and the take shows warm timings, not a 4 GB download. Clear the toasts by reloading before the take.
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
- **Output:** one continuous take per run, plus a JSON log of timestamps for each state change (upload, toggle, click, card, toast, board). The edit uses the log to place captions and zooms without scrubbing. Each event that a zoom targets also logs the element's box in CSS px (×2 for the master).
- **The script, as built** (`docs/demo/capture_demo.py <take>`):
  - A ring cursor injected with `add_init_script`. It follows real `mousemove` events and shrinks on `mousedown`.
  - Mouse glides eased over ~0.7 s in real time; Playwright's own `steps=` fire as fast as they can, so the cursor would jump.
  - Wheel scrolling in 40 px steps, because `scroll_into_view_if_needed` jumps.
  - The question edit is a JS `setSelectionRange` over the old line, then typed at 45 ms a key, then committed with ⌘+Enter, which the text area's own hint shows on screen.
- **Selectors that held** on Streamlit 1.66: keyed widgets through their `.st-key-<key>` class (`use_analysis`, `questions`, `transcribe`, `dl_md`, `dl_txt`); the visible uploader button as `[data-testid="stBaseButton-secondary"]` (`get_by_role("button")` also matches the hidden file input); a card as the innermost `stVerticalBlock` containing its heading (there is no `…BorderWrapper` test id any more); `.first` on the download buttons, where `.st-key-dl_md button` matches two.
- **Takes, 2026-11-07:** three takes of 63.2–63.5 s, 18.6–18.9 fps, no failed captures, 6.9–7.2 MB each. Every event lands within 0.35 s across takes, and frames at the same event are pixel-identical apart from a toast mid-animation. So the takes are interchangeable. **Take 4** (63.9 s, 19.4 fps) replaced them for the cut: the board and the edited questions box sat at the foot of the window (y ≈ 830 of 900), where no camera move could lift them clear of the caption. Take 4 wheel-scrolls each into the band above it (`scroll_band`, y 110–640): board 108–589, questions 141–351. The sidebar is fixed, so its toggles stay in view. The one exception is shot 4's questions box (621–831): the page has nothing to scroll until the analysis card exists.

## Edit and export

**Done (2026-11-07), scripted rather than in an editor.** `docs/demo/cut_demo.py <take>` (PyAV + Pillow) renders both formats from `master.mp4` and `log.json` in ~2.5 minutes: `demo_16x9.mp4` (1920×1080, 70.9 s, 13.9 MB) and `demo_4x5.mp4` (1080×1350, 70.9 s, 11.2 MB). H.264 high profile, `crf 18`, `+faststart`, 30 fps, silent.

- **Timeline:** the first frame is held 1 s under the title card. The take then runs **uncut at real speed** (0–63.9 s), which is what lets shot 3 say "Real time, no cuts". Then a 6 s end card, cross-faded in over 0.6 s.
- **Camera:** per-format keyframes, eased over 0.9 s. Each keyframe is a fixed region or a box the capture logged (`"@board_in_view"`), so the framing follows wherever the capture scrolled. `fit()` makes the smallest window of the output's aspect that puts the region in the part of the frame *above* the caption (the top 83% in 16:9, 79% in 4:5), clamped to the page. Zoom is capped at ~1.2× upscaling from the DSF 2 master (minimum window 800 CSS px in 16:9, 520 in 4:5), so text stays sharp.
- **Type:** Source Sans 3, the app's own face (OFL), read from the WOFF2 that Streamlit bundles. Captions are Semibold on a square Gray 100 panel with a 6 px Blue 60 rule, the theme's tokens. The title card's size is computed so its widest line fits 86% of the frame; the first 4:5 render, sized from the frame, clipped both edges.
- **Known overlap:** in the 4:5 cut at shot 4, the last two of the nine default questions sit under the caption for ~2.5 s (the page can't scroll there yet; see Capture, Takes).
- **Formats:** 1920×1080 (16:9) for X, YouTube and portfolio sites, and a 1080×1350 (4:5) crop for LinkedIn and Instagram feeds. The 4:5 version leans on zooms (shots 5–6 especially), because the full grid at phone width is too small to read.
- **Sound:** Playwright records no audio. Options:
  - silent with burned-in captions, which suits feeds that autoplay muted
  - a quiet music bed
  - optionally, 2–3 s of the meeting audio under shot 2 so viewers hear what is being transcribed

  Captions are burned in either way.
- **Captions:** high contrast, at least 4.5:1 against what they sit on (the theme was measured to that line; the video should keep it). Measured: Gray 10 on the panel 16.5:1, and 13.8:1 at worst (the 93%-opaque panel over white); muted end-card text 10.6:1. Every caption is held for its whole shot (4–16 s) with 0.25 s fades. As burned in, each checked against "What the video must not claim":

| Shot | Caption | Check |
|------|---------|-------|
| 1 | "Meeting in. Transcript, summary and action items out." / "All on your Mac." | Local by design, and makes no offline claim (the first run downloads the models) |
| 2 | "Upload audio or video, or record from the mic" | — |
| 3 | "Speech to text with Granite Speech 5.0, on device. Real time, no cuts" | No speed figure: click to card is ~1.5 s on screen (rerun, decode and VAD as well as the model's 0.56 s), so "under a second" would be false. The uncut footage is the claim |
| 4 | "Add analysis, and ask your own yes/no questions" | — |
| 5 | "Summary and action items by Granite 4.2 3B, also on device" | — |
| 6 | "Questions ranked by how likely the answer is yes" | A ranking, not a verdict; the board's own "they aren't verdicts" line is kept in frame |
| 7 | "Ask what matters to you. Only the board re-runs" | True: `run_analysis` scores the board only |
| 8 | "Export the transcript as .txt and the analysis as .md" | — |
| 9 | End card: name, "Transcription and analysis that never leave your Mac", the three model/stack names, the repo URL, "English only · Apache 2.0" | Says English only |

- **Length:** 70.9 s, inside 60–90 s, with nothing cut.

## YouTube

**Published 2026-11-08: https://youtu.be/k6xRJ1E2Nbs**, public, on a channel that is dev-focused for now. It is the 16:9 cut with sound added, a custom thumbnail and chapters. The 4:5 cut stays silent for feeds that autoplay muted. A YouTube viewer usually has the sound on, and 71 silent seconds can read as broken audio.

- **Sound** (`youtube_cut.py`): the video stream is copied, so the picture is bit-identical to the cut.
  - **Music:** "Rain Over Kyoto Station" by The Mini Vandals, from the YouTube Audio Library, filtered to "attribution not required". It was chosen by listening to four test cuts (Circuit Rush, Balearic Slide and Pulsar were the others), each mixed the same way. It starts 30 s into the track, at −26 dBFS RMS, fades in over 1.5 s and fades out across the end card. Studio's copyright check passed it.
  - **Speech:** "Morning Tom. Quick question for you." (the first 2.75 s of the meeting, ending on the pause at 3.13 s) plays when the audio player lands in shot 2, at 6.33 s, with the music ducked 12 dB. The shot-2 caption appears about 2 s before the file lands, so starting on the caption would have put the voice over an empty uploader.
  - **Timing comes from the cut, not the log:** this cut outlived its take, so `log.json` was gone. The title fade marks the upload, and the player is the largest single-frame change in the input column, since the camera's zoom changes it a little on every frame.
- **Chapters** (also printed by `youtube_cut.py`): 0:00 Upload and transcribe · 0:17 Turn on analysis · 0:30 Summary and question board · 0:41 Ask your own questions · 0:56 Export. They are read from when each burned-in caption appears. Shots are grouped because several run under YouTube's 10 s minimum per chapter.
- **Thumbnail** (`youtube_thumbnail.py`): the question board at 39.0 s beside "Meeting in. Insights out." / "Runs on your Mac · Granite · MLX". The capture's cursor ring sits on the board in every frame of shot 6, so it is painted out with the same bar 80 px to its right. The first try, copying the row above, left a 1 px seam.
- **Upload settings:** title "Meeting in, insights out: local speech AI on a Mac". The description is checked against "What the video must not claim": the first run downloads about 5 GB, the machine is the M2 Max, English only, scores rank rather than decide, and the meeting is labelled synthetic. Ten tags. Not made for kids, no paid promotion. AI use "No": the four cases YouTube lists are a real person made to say something, altered real footage, a realistic invented scene, and AI music as the focus. Text-to-speech voices that are no one, in a software demo, are none of them.
- **Verification:** a custom thumbnail needs a phone-verified channel; before that, Studio drops the upload with no error. Clickable links in the description need YouTube's *advanced* features (a selfie video, ID, or about 2 months of channel history), so until then the repo URL shows as plain text.
- **README:** a linked thumbnail from `img.youtube.com/vi/<id>/maxresdefault.jpg`, which YouTube serves as the custom thumbnail, so nothing is committed and a new thumbnail reaches the README without a commit.
- **Checked:** both scripts were run against the published files. The thumbnail is pixel-identical. The video stream is bit-identical, and the speech time and chapters match. The audio matches to float32 rounding before encoding (max 6e-8). The AAC encoder is deterministic, and that rounding is what leaves a −63 dB residue between the two encodes, 37 dB under the music.

## Reproduce

Re-shoot after a UI change, from the repo root, on macOS with Google Chrome installed. Everything a run writes goes to `docs/demo/build/`, which `.gitignore`'s `build/` already covers. A take's frames are ~1.5 GB, and they are kept so a take can be re-cut without re-capturing.

| File | What it is |
|------|------------|
| `meeting_script.txt` | The meeting, one `speaker\|line` per row: the source of truth for the audio |
| `make_meeting.py` | Voices the script with `say`, writes `build/team_meeting.wav`; `--analyse` prints what the app makes of it |
| `capture_demo.py` | Drives the app through shots 1–8 and writes `build/capture/take<N>/` |
| `cut_demo.py` | Renders `demo_16x9.mp4` and `demo_4x5.mp4` from a take |
| `youtube_cut.py` | Adds the music and the meeting line to the 16:9 cut, and prints the chapters |
| `youtube_thumbnail.py` | Makes the 1280×720 thumbnail from the 16:9 cut |

```bash
# 1. The audio, in the project venv. It imports the app, hence PYTHONPATH.
PYTHONPATH=. uv run python docs/demo/make_meeting.py --analyse

# 2. The app, in its own terminal.
uv run streamlit run streamlit_app.py --server.port 8599 --server.headless true --client.toolbarMode minimal

# 3. Capture, then cut, in an ephemeral environment pinned to the versions used here.
W="uv run --no-project --with playwright==1.64.1 --with pillow==13.0.0 --with av==19.0.2"
$W python docs/demo/capture_demo.py 1
$W python docs/demo/cut_demo.py 1

# 4. YouTube: sound and chapters in the project venv, then the thumbnail.
#    The music is the Audio Library download, which is not committed.
T=docs/demo/build/capture/take1
uv run python docs/demo/youtube_cut.py $T/demo_16x9.mp4 "Rain Over Kyoto Station - The Mini Vandals.mp3"
$W python docs/demo/youtube_thumbnail.py $T/demo_16x9.mp4
```

- **Why two environments.** `make_meeting.py` uses the app's own decoder and pipeline, and `youtube_cut.py` needs only `av` and `numpy`, so both run in the project venv and `ty` checks them. The capture, the cut and the thumbnail need Playwright or Pillow, which the app doesn't, so they get a throwaway environment instead of new project dependencies, and `pyproject.toml` excludes those three from `ty` (`[tool.ty.src]`). `ruff` and the CI format check still cover all five. An editor's ty language server doesn't read that exclusion and flags the Playwright import in `capture_demo.py`; `uv run ty check` passes.
- **The font** is the Source Sans 3 WOFF2 inside the project's `.venv` (Streamlit's), so `uv sync` has to have run.
- **Checked 2026-11-07:** from the repo, `make_meeting.py` reproduced the takes' WAV byte for byte, and one capture (63.5 s, 19.1 fps, no failed frames) cut to 70.5 s in both formats. The first capture attempt in the fresh pinned environment exited with an error its output filter hid (it matched `Error`, not uv's lowercase `error:`). An identical re-run went through, so a transient environment build is the likely cause, unconfirmed.
- **When the UI changes:** a selector that no longer matches fails loudly (Capture, Selectors that held). A layout change can leave a fixed camera region in `cut_demo.py`'s `CAMERA` pointing at the wrong place, so compare a sheet of frames before posting. The regions taken from the log (`"@event"`) follow on their own.

## Steps

1. ~~Write and voice the meeting script, then tune it against the app until the board shows a spread (Demo audio).~~ **Done.**
2. ~~Do the sharpness test take, and choose `record_video_dir` or screencast (Capture).~~ **Done:** neither; frame-grab at DSF 2.
3. ~~Write the capture script with state-based waits and the timestamp log. Do three takes and keep the cleanest.~~ **Done.** Shots 3 and 4 were swapped after the first take (Storyboard).
4. ~~Write captions and check each one against "What the video must not claim".~~ **Done** (Edit and export, Captions).
5. ~~Cut the 16:9 version, then the 4:5 crop.~~ **Done**, from take 4.
6. ~~Review on a phone, muted, before posting.~~ **Done (2026-11-07):** the 4:5 cut on an iPhone, muted and upright; every caption, the transcript, the analysis card, the board and the end card's repo URL read without zooming, and shot 4's brief overlap was accepted.

## Open questions

- Light or dark? The README screenshot is dark. Should a final shot flip the theme from the menu to show both?
- ~~Should the scripts be kept in the repo?~~ **Yes:** `docs/demo/` (Reproduce).
- Should the Record tab appear (a 2 s shot of the mic widget) even though the take uses an upload?
