"""Make the YouTube thumbnail from the 16:9 cut (docs/DEMO.md, YouTube).

Usage, from the repo root, in the same ephemeral environment as cut_demo.py:

    uv run --no-project --with pillow==13.0.0 --with av==19.0.2 \\
        python docs/demo/youtube_thumbnail.py <demo_16x9.mp4> [-o thumbnail.png]

1280x720: the question board, as the cut frames it at FRAME_S, beside a
two-line title, in the app's own face (Source Sans 3) and Carbon tokens. The
board is the thumbnail because it is the result rather than the interface, and
a column of scores from 1.00 to 0.00 still reads at thumbnail size.

The capture's cursor ring sits on the board in every frame of that shot, so it
is painted out with the same bar 80 px to its right: same row, same height,
and no label reaches that far. RING and the crop are this cut's coordinates;
after a re-shoot, check the frame (written beside the output) before posting.
"""

import argparse
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

DEMO = Path(__file__).resolve().parent
FONT_GLOB = ".venv/lib/python3*/site-packages/streamlit/static/static/media/SourceSansVF-Upright*.woff2"
INK, PANEL, MUTED, ACCENT = (
    (244, 244, 244),
    (22, 22, 22),
    (198, 198, 198),
    (15, 98, 254),
)
W, H = 1280, 720
FRAME_S = 39.0  # shot 6, the board in view
RING = (1234, 508, 1302, 604)  # the cursor ring over the 0.93 bar, frame px
PATCH_DX = 80
BOARD = (750, 30, 1766, 870)  # nine rows and the "aren't verdicts" line
TITLE = ["Meeting in.", "Insights out."]
SUB = ["Runs on your Mac", "Granite · MLX"]


def font(px: int, weight: str = "Semibold") -> ImageFont.FreeTypeFont:
    found = sorted(DEMO.parents[1].glob(FONT_GLOB))
    if not found:
        raise SystemExit("Source Sans 3 not found in .venv: run `uv sync` first")
    f = ImageFont.truetype(str(found[-1]), px)
    f.set_variation_by_name(weight)
    return f


def frame_at(video: Path, t: float) -> Image.Image:
    with av.open(str(video)) as c:
        for f in c.decode(video=0):
            if (f.time or 0.0) >= t:
                return f.to_image()
    raise SystemExit(f"{video} ends before {t} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path, help="the 16:9 cut")
    ap.add_argument(
        "-o", "--out", type=Path, default=DEMO / "build" / "youtube_thumbnail.png"
    )
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    frame = frame_at(args.video, FRAME_S).convert("RGB")
    x0, y0, x1, y1 = RING
    frame.paste(frame.crop((x0 + PATCH_DX, y0, x1 + PATCH_DX, y1)), (x0, y0))
    frame.save(args.out.with_name(f"{args.out.stem}_frame.png"))

    board = frame.crop(BOARD)
    bh = H - 110
    board = board.resize((round(board.width * bh / board.height), bh), Image.LANCZOS)
    img = Image.new("RGB", (W, H), PANEL)
    bx = W - board.width - 24
    img.paste(board, (bx, 55))

    d = ImageDraw.Draw(img)
    x, y, big = 48, 210, font(72)
    for line in TITLE:
        d.text((x, y), line, font=big, fill=INK)
        y += 86
    d.rectangle((x, y + 18, x + 80, y + 24), fill=ACCENT)
    sub = font(36, "Regular")
    for i, line in enumerate(SUB):
        d.text((x, y + 48 + 46 * i), line, font=sub, fill=MUTED)

    right = x + max(big.getlength(s) for s in TITLE)
    if right >= bx - 16:
        raise SystemExit(f"title ({right:.0f} px) runs into the board ({bx} px)")
    img.save(args.out)
    print(f"{args.out}: {img.size[0]}x{img.size[1]}")


if __name__ == "__main__":
    main()
