"""Cut the demo video from a capture take (docs/DEMO.md, Steps 4-5).

Usage, from the repo root (docs/DEMO.md, Reproduce):

    uv run --no-project --with playwright==1.64.1 --with pillow==13.0.0 --with av==19.0.2 \\
        python docs/demo/cut_demo.py <take number> [16x9|4x5 ...]

Reads docs/demo/build/capture/take<N>/master.mp4 (2940x1800, 30 fps) and <take>/log.json, and writes
<take>/demo_16x9.mp4 (1920x1080) and <take>/demo_4x5.mp4 (1080x1350): a held
title card over the empty app, the take itself uncut at real speed under a
virtual camera keyed to the log's events, burned-in captions, and an end card.
"""

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

DEMO = Path(__file__).resolve().parent
# Source Sans 3 (OFL), the app's own face, as the WOFF2 Streamlit bundles; the
# cut runs outside the project venv, so it is found in that venv's files.
FONT_GLOB = ".venv/lib/python3*/site-packages/streamlit/static/static/media/SourceSansVF-Upright*.woff2"
FPS = 30
SCALE = 2  # master pixels per CSS px
CSS_W, CSS_H = 1470, 900
HOLD_S = 1.0  # first frame held under the title card
END_S = 6.0
END_FADE_S = 0.6
MOVE_S = 0.9  # camera move duration

# Carbon Gray 100 / Gray 10, the app's own dark tokens.
INK = (244, 244, 244)
PANEL = (22, 22, 22)
MUTED = (198, 198, 198)
ACCENT = (15, 98, 254)  # Blue 60, the theme's one accent

# Shot -> caption. Checked against docs/DEMO.md, "What the video must not claim":
# no offline claim, no speed figure (the take is real time and uncut, which is
# the claim made instead), ranks not verdicts, English only on the end card.
CAPTIONS = [
    ("shot2_upload", "Upload audio or video, or record from the mic"),
    (
        "shot3_transcribe_click",
        "Speech to text with Granite Speech 5.0, on device. Real time, no cuts",
    ),
    ("shot4_analysis_toggle", "Add analysis, and ask your own yes/no questions"),
    (
        "shot5_analysis_card",
        "Summary and action items by Granite 4.2 3B, also on device",
    ),
    ("shot6_board_scroll", "Questions ranked by how likely the answer is yes"),
    ("shot7_scroll_up", "Ask what matters to you. Only the board re-runs"),
    ("shot8_downloads", "Export the transcript as .txt and the analysis as .md"),
]
TITLE = ("Meeting in. Transcript, summary", "and action items out.", "All on your Mac.")
END = [
    ("Granite Speech Intelligence", 1.0, "Semibold", INK),
    ("Transcription and analysis that never leave your Mac", 0.5, "Regular", MUTED),
    (
        "Granite Speech 5.0 TurboCTC  ·  Granite 4.2 3B  ·  MLX on Apple Silicon",
        0.4,
        "Regular",
        MUTED,
    ),
    ("github.com/darylalim/granite-speech-intelligence", 0.45, "Semibold", INK),
    ("English only  ·  Apache 2.0", 0.34, "Regular", MUTED),
]

FULL = (0, 0, CSS_W, CSS_H)
# Camera keyframes per format: ((event, offset s), region). The camera eases to
# each region over MOVE_S from that moment. A region is CSS px, or "@event" /
# ("@event", pad) for the box that event logged, so the framing follows
# wherever the capture scrolled instead of assuming it.
CAMERA = {
    "16x9": [
        (("shot1_empty_state", 0), FULL),
        (("shot2_upload", 0), (330, 60, 1430, 560)),
        (("transcript_card", -0.3), ("@transcript_card", 12)),
        (("shot4_analysis_toggle", 0), (0, 40, 940, 500)),
        (("questions_shown", -0.4), ("@questions_shown", 40)),
        (("first_analyse_shown", 0), (380, 160, 1400, 720)),
        (("first_analyse_click", 0.8), (790, 170, 1400, 720)),
        (("board_in_view", -0.2), ("@board_in_view", 12)),
        (("question_selected", -1.0), ("@question_selected", 40)),
        (("analyse_button_shown", 0), ("@analyse_button_shown", 24)),
        (("new_board_in_view", -0.2), ("@new_board_in_view", 12)),
        (("shot8_downloads", 0), (300, 60, 1470, 900)),
        (("hover_dl_txt", -0.6), ("@hover_dl_txt", 260)),
    ],
    "4x5": [
        (("shot1_empty_state", 0), (330, 60, 1050, 900)),
        (("transcript_card", -0.3), ("@transcript_card", 8)),
        (("shot4_analysis_toggle", 0), (0, 40, 720, 900)),
        (("questions_shown", -0.4), ("@questions_shown", 24)),
        (("first_analyse_shown", 0), (380, 160, 910, 720)),
        (("first_analyse_click", 0.8), (795, 175, 1395, 900)),
        (("board_in_view", -0.2), ("@board_in_view", 8)),
        (("question_selected", -1.0), ("@question_selected", 24)),
        (("analyse_button_shown", 0), ("@analyse_button_shown", 16)),
        (("new_board_in_view", -0.2), ("@new_board_in_view", 8)),
        (("shot8_downloads", 0), (790, 170, 1400, 900)),
        (("hover_dl_txt", -0.6), ("@hover_dl_txt", 200)),
    ],
}


def logged_region(spec, boxes: dict) -> tuple:
    if isinstance(spec, tuple) and isinstance(spec[0], str):
        name, pad = spec
    elif isinstance(spec, str):
        name, pad = spec, 0
    else:
        return spec
    b = boxes[name.lstrip("@")]
    return (
        b["x"] - pad,
        b["y"] - pad,
        b["x"] + b["width"] + pad,
        b["y"] + b["height"] + pad,
    )


@dataclass
class Format:
    name: str
    w: int
    h: int
    min_css_w: float  # caps the zoom: at most ~1.2x upscaling from the master
    caption_px: int
    caption_bottom: float  # caption box bottom, as a fraction of height
    content_frac: float  # the part of the frame above the caption band


FORMATS = {
    "16x9": Format("16x9", 1920, 1080, 800, 46, 0.93, 0.83),
    "4x5": Format("4x5", 1080, 1350, 520, 50, 0.94, 0.79),
}


def _font_path() -> Path:
    found = sorted(DEMO.parents[1].glob(FONT_GLOB))
    if not found:
        raise SystemExit("Source Sans 3 not found in .venv: run `uv sync` first")
    return found[-1]


FONT = _font_path()


def font(px: int, weight: str = "Semibold") -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONT), px)
    f.set_variation_by_name(weight)
    return f


def ease(k: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(max(k, 0.0), 1.0))


def fit(region: tuple, fmt: Format) -> tuple[float, float, float]:
    """Smallest window of the format's aspect that shows region above the
    caption band (the top content_frac of the frame), clamped to the frame.
    Returns (cx, cy, width) in CSS px. Near the page's bottom edge the clamp
    wins, and the caption may still overlap the region."""
    x0, y0, x1, y1 = region
    aspect = fmt.w / fmt.h
    w = max(x1 - x0, (y1 - y0) * aspect / fmt.content_frac, fmt.min_css_w)
    w = min(w, CSS_W, CSS_H * aspect)
    h = w / aspect
    cx = min(max((x0 + x1) / 2, w / 2), CSS_W - w / 2)
    cy = (y0 + y1) / 2 + h * (1 - fmt.content_frac) / 2
    cy = min(max(cy, h / 2), CSS_H - h / 2)
    return cx, cy, w


def wrap(text: str, f: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if f.getlength(trial) <= max_w or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return [*lines, line]


def caption_image(text: str, fmt: Format) -> Image.Image:
    """A square-cornered Gray 100 panel with a Blue 60 rule, text in Gray 10:
    16.5:1 on the panel and 13.8:1 at worst (the 93%-opaque panel over white),
    against the 4.5:1 the theme was measured to."""
    f = font(fmt.caption_px)
    lines = wrap(text, f, fmt.w * 0.82)
    asc, desc = f.getmetrics()
    lh = round((asc + desc) * 1.12)
    pad_x, pad_y, rule = round(fmt.caption_px * 0.6), round(fmt.caption_px * 0.42), 6
    tw = max(f.getlength(line) for line in lines)
    img = Image.new("RGBA", (round(tw + 2 * pad_x + rule), lh * len(lines) + 2 * pad_y))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, img.width, img.height), fill=(*PANEL, 238))
    d.rectangle((0, 0, rule - 1, img.height), fill=(*ACCENT, 255))
    for i, line in enumerate(lines):
        d.text((rule + pad_x, pad_y + i * lh), line, font=f, fill=(*INK, 255))
    return img


def title_overlay(fmt: Format) -> Image.Image:
    img = Image.new("RGBA", (fmt.w, fmt.h), (*PANEL, 205))
    d = ImageDraw.Draw(img)
    # Sized from the text, not the frame: the largest size whose widest line
    # fits 86% of the width (the 4:5 cut clipped both edges when it wasn't).
    px = round(fmt.w * 0.074)
    while max(font(px).getlength(line) for line in TITLE[:2]) > fmt.w * 0.86:
        px -= 2
    big = font(px)
    sub = font(round(px * 0.62), "Regular")
    rows = [
        (TITLE[0], big, INK),
        (TITLE[1], big, INK),
        ("", sub, INK),
        (TITLE[2], sub, MUTED),
    ]
    heights = [sum(f.getmetrics()) * (0.4 if not t else 1.15) for t, f, _ in rows]
    y = (fmt.h - sum(heights)) / 2
    for (t, f, color), h in zip(rows, heights):
        if t:
            d.text(((fmt.w - f.getlength(t)) / 2, y), t, font=f, fill=(*color, 255))
        y += h
    return img


def end_card(fmt: Format) -> Image.Image:
    img = Image.new("RGB", (fmt.w, fmt.h), PANEL)
    d = ImageDraw.Draw(img)
    base = fmt.w * (0.05 if fmt.w > fmt.h else 0.075)
    rows = []
    for text, size, weight, color in END:
        f = font(round(base * size), weight)
        lines = wrap(text, f, fmt.w * 0.86)
        rows += [(line, f, color) for line in lines] + [("", f, color)]
    heights = [sum(f.getmetrics()) * (0.55 if not t else 1.1) for t, f, _ in rows[:-1]]
    y = (fmt.h - sum(heights)) / 2
    for (t, f, color), h in zip(rows, heights):
        if t:
            d.text(((fmt.w - f.getlength(t)) / 2, y), t, font=f, fill=color)
        y += h
    bar_w = round(fmt.w * 0.08)
    d.rectangle(
        (
            (fmt.w - bar_w) // 2,
            round(y + base * 0.2),
            (fmt.w + bar_w) // 2,
            round(y + base * 0.2) + 5,
        ),
        fill=ACCENT,
    )
    return img


def render(take: Path, fmt: Format) -> Path:
    log = json.loads((take / "log.json").read_text())
    at = {e["event"]: e["t"] for e in log["events"]}
    src_dur = log["duration_s"]
    boxes = {e["event"]: e["box_css"] for e in log["events"] if "box_css" in e}
    keys = sorted(
        (at[name] + off, fit(logged_region(region, boxes), fmt))
        for (name, off), region in CAMERA[fmt.name]
    )
    starts = [(at[name], caption_image(text, fmt)) for name, text in CAPTIONS]
    ends = [s for s, _ in starts[1:]] + [at["end"] + 0.6]
    captions = [(s + 0.2, e - 0.15, img) for (s, img), e in zip(starts, ends)]
    title = title_overlay(fmt)
    title_end = HOLD_S + at["shot2_upload"]
    card = end_card(fmt)
    total = HOLD_S + src_dur + END_S

    def camera(t: float) -> tuple[float, float, float]:
        cur = keys[0][1]
        for start, target in keys[1:]:
            if t <= start:
                break
            k = ease((t - start) / MOVE_S)
            cur = tuple(a + (b - a) * k for a, b in zip(cur, target))
        return cur

    out = take / f"demo_{fmt.name}.mp4"
    src = av.open(str(take / "master.mp4"))
    frames = src.decode(video=0)
    frame, frame_i = next(frames).to_image(), 0
    last_view = None
    with av.open(str(out), "w", options={"movflags": "+faststart"}) as dst:
        s = dst.add_stream("libx264", rate=FPS)
        s.width, s.height, s.pix_fmt = fmt.w, fmt.h, "yuv420p"
        s.options = {"crf": "18", "preset": "medium", "profile": "high"}
        for n in range(round(total * FPS)):
            t_out = n / FPS
            t_src = min(max(t_out - HOLD_S, 0.0), src_dur)
            if t_out < HOLD_S + src_dur:
                want = round(t_src * FPS)
                while frame_i < want:
                    try:
                        frame, frame_i = next(frames).to_image(), frame_i + 1
                    except StopIteration:
                        break
                cx, cy, w = camera(t_src)
                h = w * fmt.h / fmt.w
                box = tuple(
                    round(v * SCALE)
                    for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                )
                view = (
                    frame.crop(box)
                    .resize((fmt.w, fmt.h), Image.LANCZOS)
                    .convert("RGBA")
                )
                if t_out < title_end:
                    a = 1.0 - ease((t_out - (title_end - 0.5)) / 0.5)
                    view.alpha_composite(
                        Image.blend(Image.new("RGBA", title.size), title, a)
                        if a < 1
                        else title
                    )
                for c0, c1, img in captions:
                    if c0 <= t_src <= c1:
                        a = min(ease((t_src - c0) / 0.25), ease((c1 - t_src) / 0.25))
                        if a < 1:
                            img = img.copy()
                            img.putalpha(
                                img.getchannel("A").point(lambda v, a=a: round(v * a))
                            )
                        view.alpha_composite(
                            img,
                            (
                                (fmt.w - img.width) // 2,
                                round(fmt.h * fmt.caption_bottom) - img.height,
                            ),
                        )
                last_view = view.convert("RGB")
                rgb = last_view
            else:
                k = ease((t_out - HOLD_S - src_dur) / END_FADE_S)
                rgb = Image.blend(last_view, card, k) if k < 1 else card
            for pkt in s.encode(av.VideoFrame.from_image(rgb)):
                dst.mux(pkt)
        for pkt in s.encode():
            dst.mux(pkt)
    src.close()
    print(f"{out}: {total:.1f}s, {out.stat().st_size / 1e6:.1f} MB")
    return out


def main() -> None:
    take = DEMO / "build" / "capture" / f"take{int(sys.argv[1])}"
    names = sys.argv[2:] or list(FORMATS)
    for name in names:
        render(take, FORMATS[name])


if __name__ == "__main__":
    main()
