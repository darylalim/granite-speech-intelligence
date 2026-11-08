"""Add the YouTube sound to the silent 16:9 cut and print its chapters
(docs/DEMO.md, YouTube).

Usage, from the repo root, in the project venv (av and numpy are app
dependencies):

    uv run python docs/demo/youtube_cut.py <demo_16x9.mp4> <music file> [-o out.mp4]

The video stream is copied, not re-encoded, so the picture is bit-identical to
the cut. The sound is built here:

- the music from MUSIC_START_S into the track, at MUSIC_DBFS RMS, faded in over
  1.5 s and out across the end card;
- the meeting's first two sentences (build/team_meeting.wav, from
  make_meeting.py) when the audio player lands in shot 2, with the music
  ducked DUCK_DB beneath them.

Neither moment comes from a log: a cut can outlive its take (log.json lives
with the take's frames), so both are found in the cut itself. The title card is
a still until its fade, which ends at the upload; the player lands in a single
frame, where the camera's zoom is eased over many. The chapters are found the
same way, from the burned-in captions, and checked against YouTube's rules.
"""

import argparse
from pathlib import Path

import av
import numpy as np

DEMO = Path(__file__).resolve().parent
MEETING = DEMO / "build" / "team_meeting.wav"
SR = 48_000
MUSIC_START_S = 30.0  # past the intro, which is not what plays under a cut
MUSIC_DBFS = -26.0  # RMS; under the captions, not over them
SPEECH_DBFS = -18.0
DUCK_DB = -12.0
FADE_IN_S = 1.5
END_S = 6.0  # cut_demo.py's END_S: the end card the music fades across
# "Morning Tom. Quick question for you." ends in silence at 3.13 s.
SPEECH_FROM_S, SPEECH_LEN_S = 0.5, 2.75
SPEECH_DELAY_S = 0.2  # after the player lands

# Chapters, by the caption that opens them (index into cut_demo.py's
# CAPTIONS; None is 0:00). Several shots run under 10 s, YouTube's minimum
# chapter, so shots are grouped.
CHAPTERS = [
    (None, "Upload and transcribe"),
    (2, "Turn on analysis"),
    (3, "Summary and question board"),
    (5, "Ask your own questions"),
    (6, "Export"),
]
N_CAPTIONS = 7


def decode(path: Path, layout: str) -> np.ndarray:
    """(channels, samples) float32 at SR."""
    with av.open(str(path)) as c:
        rs = av.AudioResampler(format="fltp", layout=layout, rate=SR)
        chunks = [
            f.to_ndarray() for frame in c.decode(audio=0) for f in rs.resample(frame)
        ]
        chunks += [f.to_ndarray() for f in rs.resample(None)]
    return np.concatenate(chunks, axis=1)


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12))


def ramp(n: int, a: float, b: float) -> np.ndarray:
    return a + (b - a) * (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n)))


def gray_frames(video: Path):
    """(time s, full-size 8-bit luma) for every frame."""
    with av.open(str(video)) as c:
        for f in c.decode(video=0):
            yield float(f.time or 0.0), f.reformat(format="gray").to_ndarray()


def duration(video: Path) -> float:
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        if s.duration is None or s.time_base is None:
            raise SystemExit(f"{video}: no stream duration")
        return float(s.duration * s.time_base)


def upload_time(video: Path) -> float:
    """First frame that differs from frame 0 starts the title's 0.5 s fade,
    which ends at the upload."""
    first = None
    for t, g in gray_frames(video):
        small = g[::10, ::10].astype(np.float32)
        if first is None:
            first = small
        elif np.abs(small - first).mean() > 1.0:
            return t + 0.5
    raise SystemExit("the title card never fades")


def player_lands(video: Path, after: float, before: float = 10.0) -> float:
    """The largest one-frame change in the input column during shot 2."""
    best, prev = (0.0, after), None
    for t, g in gray_frames(video):
        if t < after:
            continue
        if t > before:
            break
        col = g[300:880, 100:800].astype(np.float32)
        if prev is not None:
            best = max(best, (float(np.abs(col - prev).mean()), t))
        prev = col
    return best[1]


def caption_starts(video: Path) -> list[float]:
    """When each burned-in caption appears: the caption band fills with bright
    text. A swap fades out and in, which can read as a few flickers when the
    new caption lands over bright page content, so starts within 1 s merge."""
    starts: list[float] = []
    on = False
    for t, g in gray_frames(video):
        now = int((g[900:1000, 300:1620] > 150).sum()) > 2000
        if now and not on and (not starts or t - starts[-1] > 1.0):
            starts.append(t)
        on = now
    return starts


def chapters(video: Path, total: float) -> list[str]:
    starts = caption_starts(video)
    if len(starts) != N_CAPTIONS:
        raise SystemExit(
            f"found {len(starts)} captions, expected {N_CAPTIONS}: {starts}"
        )
    marks = [0 if i is None else int(starts[i]) for i, _ in CHAPTERS]
    ends = [*marks[1:], int(total)]
    for (_, title), m, e in zip(CHAPTERS, marks, ends, strict=True):
        if e - m < 10:
            raise SystemExit(f"chapter {title!r} is {e - m} s; YouTube needs 10")
    return [
        f"{m // 60}:{m % 60:02d} {title}"
        for (_, title), m in zip(CHAPTERS, marks, strict=True)
    ]


def speech_clip() -> np.ndarray:
    if not MEETING.exists():
        raise SystemExit(f"{MEETING} missing: run make_meeting.py first")
    meet = decode(MEETING, "mono")[0]
    a = round(SPEECH_FROM_S * SR)
    clip = meet[a : a + round(SPEECH_LEN_S * SR)].copy()
    clip *= 10 ** ((SPEECH_DBFS - rms_db(clip)) / 20)
    fo = round(0.1 * SR)
    clip[-fo:] *= ramp(fo, 1, 0)
    return clip


def mix(
    music_path: Path, total: float, speech: np.ndarray, t_speech: float
) -> np.ndarray:
    n = round(total * SR)
    s0 = round(MUSIC_START_S * SR)
    music = decode(music_path, "stereo")[:, s0 : s0 + n]
    if music.shape[1] < n:
        raise SystemExit(
            f"{music_path.name}: under {total:.0f} s from {MUSIC_START_S} s"
        )
    music *= 10 ** ((MUSIC_DBFS - rms_db(music)) / 20)

    gain = np.ones(n, np.float32)
    fi = round(FADE_IN_S * SR)
    gain[:fi] *= ramp(fi, 0, 1)
    fo0 = round((total - END_S) * SR)
    gain[fo0:] *= ramp(n - fo0, 1, 0)

    duck = np.ones(n, np.float32)
    r = round(0.4 * SR)
    d0, d1 = round(t_speech * SR) - r, round((t_speech + SPEECH_LEN_S) * SR)
    lo = 10 ** (DUCK_DB / 20)
    duck[d0 : d0 + r] = ramp(r, 1, lo)
    duck[d0 + r : d1] = lo
    duck[d1 : d1 + r] = ramp(r, lo, 1)

    out = music * gain * duck
    a = round(t_speech * SR)
    out[:, a : a + speech.shape[0]] += speech
    peak = float(np.abs(out).max())
    if peak > 0.98:
        out *= 0.98 / peak
    return out.astype(np.float32)


def write(video: Path, audio: np.ndarray, out: Path) -> None:
    """Copy the video packets and mux AAC alongside, interleaved by time."""
    with (
        av.open(str(video)) as src,
        av.open(str(out), "w", options={"movflags": "+faststart"}) as dst,
    ):
        vin = src.streams.video[0]
        vout = dst.add_stream_from_template(vin)
        aout = dst.add_stream("aac", rate=SR, layout="stereo")
        aout.bit_rate = 192_000
        apkts = []
        for i in range(0, audio.shape[1], 1024):
            fr = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(audio[:, i : i + 1024]),
                format="fltp",
                layout="stereo",
            )
            fr.sample_rate, fr.pts = SR, i
            apkts += aout.encode(fr)
        apkts += aout.encode(None)

        def at(p: av.Packet) -> float:
            return float((p.dts or 0) * (p.time_base or 0))

        ai = 0
        for vp in src.demux(vin):
            if vp.dts is None:
                continue
            while ai < len(apkts) and at(apkts[ai]) <= at(vp):
                dst.mux(apkts[ai])
                ai += 1
            vp.stream = vout
            dst.mux(vp)
        for p in apkts[ai:]:
            dst.mux(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path, help="the silent 16:9 cut")
    ap.add_argument("music", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    args = ap.parse_args()
    out = args.out or args.video.with_name(f"{args.video.stem}_youtube.mp4")

    total = duration(args.video)
    t_speech = player_lands(args.video, upload_time(args.video)) + SPEECH_DELAY_S
    write(args.video, mix(args.music, total, speech_clip(), t_speech), out)
    print(f"{out}: {total:.2f} s, speech at {t_speech:.2f} s")
    print("\nChapters, for the description:")
    print("\n".join(chapters(args.video, total)))


if __name__ == "__main__":
    main()
