"""Capture the demo video's app shots (docs/DEMO.md, Storyboard shots 1-8).

Usage, from the repo root, with the app running on port 8599 (docs/DEMO.md,
Reproduce):

    uv run --no-project --with playwright==1.64.1 --with pillow==13.0.0 --with av==19.0.2 \\
        python docs/demo/capture_demo.py <take number> [--no-warm]

Drives the real app at 1470x900, DSF 2, dark theme, while a CDP
captureScreenshot loop grabs 2940x1800 frames (~20 fps) to disk. Afterwards
the frames are resampled to constant 30 fps and encoded with libx264. Writes
docs/demo/build/capture/take<N>/{frames/, master.mp4, log.json}; log.json holds every state
change's time in the master plus the boxes the editor zooms to."""

import asyncio
import base64
import itertools
import json
import math
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import av
from PIL import Image
from playwright.async_api import (
    Error as PlaywrightError,
    Locator,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

URL = "http://localhost:8599"
DEMO = Path(__file__).resolve().parent
BUILD = DEMO / "build"  # git-ignored
WAV = BUILD / "team_meeting.wav"  # from make_meeting.py
VIEWPORT = {"width": 1470, "height": 900}
DSF = 2
FPS = 30
OLD_QUESTION = "Does the speaker mention a place?"
NEW_QUESTION = "Is a bug or problem discussed?"
BOARD_CAPTION = "Sorted by P(yes), highest first."

# A soft ring that follows real mouse events; Playwright draws no cursor.
CURSOR_JS = """
addEventListener('DOMContentLoaded', () => {
  const d = document.createElement('div');
  Object.assign(d.style, {
    position: 'fixed', left: '0', top: '0', width: '24px', height: '24px',
    margin: '-12px 0 0 -12px', borderRadius: '50%', pointerEvents: 'none',
    background: 'rgba(255,255,255,0.25)', border: '2px solid rgba(255,255,255,0.9)',
    boxShadow: '0 0 0 1px rgba(0,0,0,0.5)', zIndex: 2147483647,
    transform: 'translate(-100px,-100px)', transition: 'scale 120ms ease',
  });
  document.body.appendChild(d);
  addEventListener('mousemove', e => {
    d.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
  }, true);
  addEventListener('mousedown', () => { d.style.scale = '0.7'; }, true);
  addEventListener('mouseup', () => { d.style.scale = '1'; }, true);
});
"""


class Take:
    def __init__(self, page: Page, out: Path) -> None:
        self.page = page
        self.out = out
        self.events: list[dict] = []
        self.pos = (VIEWPORT["width"] * 0.6, VIEWPORT["height"] * 0.55)

    def log(self, name: str, box: dict | None = None) -> None:
        event = {"event": name, "t": time.monotonic()}
        if box:
            event["box_css"] = {k: round(v, 1) for k, v in box.items()}
        self.events.append(event)
        print(f"  {name}")

    async def beat(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    async def glide(self, x: float, y: float, seconds: float = 0.7) -> None:
        """Ease-in-out mouse move in real time, so the ring visibly travels."""
        x0, y0 = self.pos
        steps = max(1, round(seconds * 60))
        for i in range(1, steps + 1):
            k = 0.5 - 0.5 * math.cos(math.pi * i / steps)
            await self.page.mouse.move(x0 + (x - x0) * k, y0 + (y - y0) * k)
            await asyncio.sleep(seconds / steps)
        self.pos = (x, y)

    async def glide_to(self, target: Locator, seconds: float = 0.7) -> None:
        box = await target.bounding_box()
        assert box, target
        await self.glide(
            box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, seconds
        )

    async def click(self, target: Locator) -> None:
        await self.glide_to(target)
        await self.beat(0.25)
        await self.page.mouse.down()
        await asyncio.sleep(0.08)
        await self.page.mouse.up()

    async def scroll_until(self, target: Locator, bottom_margin: float = 60) -> None:
        """Wheel-scroll the main area in small steps until target is in view."""
        for _ in range(200):
            box = await target.bounding_box()
            if box and box["y"] + box["height"] <= VIEWPORT["height"] - bottom_margin:
                return
            await self.page.mouse.wheel(0, 40)
            await asyncio.sleep(1 / 30)
        raise RuntimeError("scrolled 200 steps without reaching the target")

    async def scroll_to_top(self) -> None:
        main = self.page.locator('[data-testid="stMain"]')
        for _ in range(200):
            if await main.evaluate("m => m.scrollTop") <= 0:
                return
            await self.page.mouse.wheel(0, -60)
            await asyncio.sleep(1 / 30)

    async def scroll_band(
        self, target: Locator, top: float = 110, bottom: float = 640
    ) -> None:
        """Wheel-scroll until target sits between top and bottom (CSS px), so
        it is clear of the caption band the cut puts at the frame's foot. If
        target is taller than the band its top wins; if the page can't
        scroll further, stop where it is."""
        last = None
        for _ in range(200):
            box = await target.bounding_box()
            assert box, target
            if box["y"] < top:
                step = -40
            elif box["y"] + box["height"] > bottom and box["y"] - 40 >= top:
                step = 40
            else:
                return
            if last is not None and abs(box["y"] - last) < 1:
                return  # the page has no more scroll in that direction
            last = box["y"]
            await self.page.mouse.wheel(0, step)
            await asyncio.sleep(1 / 30)


def union(*boxes: dict) -> dict:
    x0 = min(b["x"] for b in boxes)
    y0 = min(b["y"] for b in boxes)
    x1 = max(b["x"] + b["width"] for b in boxes)
    y1 = max(b["y"] + b["height"] for b in boxes)
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


async def box_of(locator: Locator) -> dict:
    box = await locator.bounding_box()
    assert box, locator
    return box


def card(page: Page, heading: str) -> Locator:
    """The bordered card that holds a given subheader."""
    return (
        page.locator('[data-testid="stVerticalBlock"]')
        .filter(has=page.get_by_role("heading", name=heading, exact=True))
        .last
    )


async def shots(t: Take) -> None:
    page = t.page
    # 1 -- empty state, held under the title card.
    t.log("shot1_empty_state")
    await t.beat(3.0)

    # 2 -- upload.
    t.log("shot2_upload")
    uploader = page.locator('[data-testid="stFileUploader"]')
    await t.glide_to(uploader.locator('[data-testid="stBaseButton-secondary"]'))
    await t.beat(0.4)
    await uploader.locator("input[type=file]").set_input_files(WAV)
    await page.locator('[data-testid="stAudio"]').wait_for()
    t.log("player_shown", await box_of(page.locator('[data-testid="stAudio"]')))
    await t.beat(2.5)

    # 3 -- transcribe, analysis still off, so the transcript card is on screen.
    t.log("shot3_transcribe_click")
    await t.click(page.locator(".st-key-transcribe button"))
    await page.get_by_text("Transcription complete").wait_for()
    t.log("transcript_card", await box_of(card(page, "Transcription")))
    await t.glide(VIEWPORT["width"] * 0.75, VIEWPORT["height"] * 0.6, 1.0)
    await t.beat(3.5)

    # 4 -- analysis on after the run: the app offers Analyse, not a re-transcribe.
    t.log("shot4_analysis_toggle")
    await t.click(page.locator(".st-key-use_analysis label"))
    expander = page.get_by_text("Analysis questions", exact=True)
    await expander.wait_for()
    await t.beat(1.0)
    await t.click(expander)
    questions = page.locator(".st-key-questions textarea")
    await questions.wait_for()
    await t.beat(0.5)
    await t.scroll_band(questions)
    t.log("questions_shown", await box_of(questions))
    await t.beat(2.5)
    await t.scroll_to_top()
    analyse = page.get_by_role("button", name="Analyse")
    t.log("first_analyse_shown", await box_of(analyse))
    await t.beat(0.4)
    await t.click(analyse)
    t.log("first_analyse_click")
    try:
        await page.get_by_text("Summarising").wait_for(timeout=5000)
        t.log("summarising")
    except PlaywrightTimeoutError:
        t.log("summarising_missed")

    # 5 -- analysis card.
    await page.get_by_text("Analysis complete").wait_for(timeout=120_000)
    analysis = card(page, "Analysis")
    t.log("shot5_analysis_card", await box_of(analysis))
    await t.glide(VIEWPORT["width"] * 0.75, VIEWPORT["height"] * 0.45, 1.2)
    await t.beat(4.0)

    # 6 -- the board, scrolled clear of the caption band, its own caption included.
    t.log("shot6_board_scroll")
    caption = page.get_by_text(BOARD_CAPTION)
    board = page.locator('[data-testid="stProgress"]')
    await t.scroll_band(caption, top=0, bottom=620)
    t.log("board_in_view", union(await box_of(board.first), await box_of(caption)))
    await t.beat(4.0)

    # 7 -- replace the place question, analyse the board only.
    t.log("shot7_scroll_up")
    await t.scroll_band(questions)
    await t.beat(0.6)
    await t.click(questions)
    await t.beat(0.4)
    found = await questions.evaluate(
        """(el, q) => {
            const i = el.value.indexOf(q);
            if (i >= 0) { el.focus(); el.setSelectionRange(i, i + q.length); }
            return i;
        }""",
        OLD_QUESTION,
    )
    assert found >= 0, "the place question is not in the box"
    t.log("question_selected", await box_of(questions))
    await t.beat(0.8)
    await page.keyboard.type(NEW_QUESTION, delay=45)
    await t.beat(0.5)
    await page.keyboard.press(
        "Meta+Enter"
    )  # Streamlit commits a text_area on Cmd+Enter
    await analyse.wait_for()
    t.log(
        "analyse_button_shown",
        union(await box_of(questions), await box_of(analyse)),
    )
    await t.beat(1.0)
    await t.click(analyse)
    t.log("analyse_click")
    new_row = board.filter(has_text=NEW_QUESTION)
    await new_row.wait_for(timeout=60_000)
    t.log("board_rescored")
    await t.beat(0.8)
    await t.scroll_band(caption, top=0, bottom=620)
    t.log("new_board_in_view", union(await box_of(board.first), await box_of(caption)))
    await t.beat(3.5)

    # 8 -- the two downloads.
    t.log("shot8_downloads")
    await t.scroll_to_top()
    for key in ("dl_md", "dl_txt"):
        button = page.locator(f".st-key-{key} button").first
        await t.scroll_band(button)
        await t.glide_to(button)
        t.log(f"hover_{key}", await box_of(button))
        await t.beat(1.6)
    t.log("end")
    await t.beat(1.0)


async def warm_up(browser) -> None:
    """Run the whole flow once so both models are in the server's cache."""
    ctx = await browser.new_context(viewport=VIEWPORT)
    page = await ctx.new_page()
    await page.goto(URL)
    await page.locator(
        '[data-testid="stFileUploader"] input[type=file]'
    ).set_input_files(WAV)
    await page.locator('[data-testid="stAudio"]').wait_for()
    await page.locator(".st-key-use_analysis label").click()
    await page.get_by_text("Analysis questions", exact=True).wait_for()
    await page.locator(".st-key-transcribe button").click()
    await page.get_by_text("Analysis complete").wait_for(timeout=600_000)
    await ctx.close()


async def capture(take: int, warm: bool) -> Path:
    out = BUILD / "capture" / f"take{take}"
    shutil.rmtree(out, ignore_errors=True)
    (out / "frames").mkdir(parents=True)
    frames: list[tuple[float, Path]] = []
    failed = 0
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome")
        if warm:
            t0 = time.perf_counter()
            await warm_up(browser)
            print(f"warm-up: {time.perf_counter() - t0:.1f}s")
        ctx = await browser.new_context(
            viewport=VIEWPORT, device_scale_factor=DSF, color_scheme="dark"
        )
        await ctx.add_init_script(CURSOR_JS)
        page = await ctx.new_page()
        await page.goto(URL)
        await page.locator('[data-testid="stFileUploader"]').wait_for()
        await page.get_by_text("Upload or record audio to get started.").wait_for()
        take_ = Take(page, out)
        await page.mouse.move(*take_.pos)
        cdp = await ctx.new_cdp_session(page)
        stop = asyncio.Event()
        clip = {"x": 0, "y": 0, **VIEWPORT, "scale": DSF}

        async def grab() -> None:
            nonlocal failed
            nxt = time.monotonic()
            while not stop.is_set():
                try:
                    shot = await cdp.send(
                        "Page.captureScreenshot",
                        {"format": "png", "optimizeForSpeed": True, "clip": clip},
                    )
                except PlaywrightError:  # a capture mid-rerender; skip the frame
                    failed += 1
                else:
                    path = out / "frames" / f"{len(frames):05d}.png"
                    path.write_bytes(base64.b64decode(shot["data"]))
                    frames.append((time.monotonic(), path))
                nxt += 1 / FPS
                await asyncio.sleep(max(0, nxt - time.monotonic()))

        grabber = asyncio.create_task(grab())
        await asyncio.sleep(0.2)
        try:
            await shots(take_)
        finally:
            stop.set()
            await grabber
            await browser.close()

    times = [t for t, _ in frames]
    gaps = sorted(b - a for a, b in itertools.pairwise(times))
    span = times[-1] - times[0]
    print(
        f"frames: {len(frames)} over {span:.1f}s = {len(frames) / span:.1f} fps, "
        f"max gap {gaps[-1] * 1000:.0f} ms, failed {failed}"
    )
    start = times[0]
    log = {
        "viewport_css": VIEWPORT,
        "scale": DSF,
        "fps": FPS,
        "duration_s": round(span, 2),
        "events": [{**e, "t": round(e["t"] - start, 2)} for e in take_.events],
    }
    (out / "log.json").write_text(json.dumps(log, indent=1))

    master = out / "master.mp4"
    with av.open(str(master), "w") as c:
        s = c.add_stream("libx264", rate=FPS)
        s.width, s.height = VIEWPORT["width"] * DSF, VIEWPORT["height"] * DSF
        s.pix_fmt = "yuv420p"
        s.options = {"crf": "14", "preset": "medium", "tune": "stillimage"}
        i = 0
        for k in range(round(span * FPS)):
            tick = start + k / FPS
            while i + 1 < len(frames) and frames[i + 1][0] <= tick:
                i += 1
            img = Image.open(frames[i][1]).convert("RGB")
            for pkt in s.encode(av.VideoFrame.from_image(img)):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    print(f"{master}: {master.stat().st_size / 1e6:.1f} MB")
    return out


def main() -> None:
    take = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for _ in range(60):
        try:
            with urllib.request.urlopen(URL + "/_stcore/health", timeout=1):
                break
        except OSError:
            time.sleep(1)
    else:
        raise SystemExit(
            f"No app at {URL} after 60 s. Start it first (docs/DEMO.md, Reproduce)."
        )
    asyncio.run(capture(take, warm="--no-warm" not in sys.argv))


if __name__ == "__main__":
    main()
