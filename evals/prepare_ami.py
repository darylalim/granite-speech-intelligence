"""Fetch the AMI data the evaluation needs and transcribe its excerpts.

Run from the repo root: ``uv run python -m evals.prepare_ami``. Everything it
writes goes under ``evals/data/`` (git-ignored): the corpus is CC BY 4.0, but
its audio and transcripts are never committed here, only the labels.

The upstream server (groups.inf.ed.ac.uk) is often unreachable, so this reads
from a Hugging Face mirror that redistributes the official archive unchanged:
the v1.6.2 manual annotations and the Mix-Headset audio of the 16-meeting
AMI-SDM evaluation split, which is where all 20 labelled excerpts come from.
"""

import io
import json
import time
import urllib.request
import wave
import zipfile
from pathlib import Path
from typing import cast

from streamlit.runtime.uploaded_file_manager import UploadedFile

from evals.ami import Meeting

EVALS_DIR = Path(__file__).parent
DATA_DIR = EVALS_DIR / "data" / "ami"
MANUAL_DIR = DATA_DIR / "manual"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
LABELS_PATH = EVALS_DIR / "labels.json"

MIRROR = "https://huggingface.co/datasets/FluidInference/ami-corpus-mirror/resolve/main"
ANNOTATIONS_ZIP = "ami_public_manual_1.6.2.zip"


def load_labels() -> dict:
    return json.loads(LABELS_PATH.read_text())


def _fetch(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    print(f"fetching {url}")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    partial.rename(path)


def fetch(meetings: set[str]) -> None:
    archive = DATA_DIR / ANNOTATIONS_ZIP
    _fetch(f"{MIRROR}/annotations/{ANNOTATIONS_ZIP}", archive)
    if not (MANUAL_DIR / "words").exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(MANUAL_DIR)
    for meeting in sorted(meetings):
        name = f"{meeting}.Mix-Headset.wav"
        _fetch(f"{MIRROR}/sdm/{name}", AUDIO_DIR / name)


def excerpt_wav(meeting: str, start: float, end: float) -> bytes:
    """The excerpt's span of the meeting's Mix-Headset WAV, as WAV bytes."""
    with wave.open(str(AUDIO_DIR / f"{meeting}.Mix-Headset.wav"), "rb") as source:
        rate = source.getframerate()
        source.setpos(round(start * rate))
        frames = source.readframes(round((end - start) * rate))
        params = source.getparams()
    out = io.BytesIO()
    with wave.open(out, "wb") as sink:
        sink.setparams(params)
        sink.writeframes(frames)
    return out.getvalue()


def plain_text(transcript: str) -> str:
    """`run_pipeline`'s transcript without the `[M:SS - M:SS]` line prefixes.

    The analysis is fed segment text only (docs/PLAN.md, Phase 1): timestamps cost
    tokens and invite made-up ones.
    """
    return " ".join(line.split("] ", 1)[-1] for line in transcript.splitlines()).strip()


def transcribe(excerpts: list[dict]) -> None:
    # Imported here so `fetch` and the label tooling never pay for torch/MLX.
    import mlx.core as mx

    import streamlit_app as app

    mx.set_cache_limit(app.MLX_CACHE_LIMIT_BYTES)
    model = app.load_model(app.MODEL_ID, revision=app.MODEL_REVISION)
    vad_model = app.load_vad_model()
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    for excerpt in excerpts:
        human = TRANSCRIPTS_DIR / f"{excerpt['id']}.human.txt"
        asr = TRANSCRIPTS_DIR / f"{excerpt['id']}.asr.txt"
        meeting = Meeting(MANUAL_DIR, excerpt["meeting"])
        human.write_text(meeting.transcript(excerpt["start"], excerpt["end"]) + "\n")
        if asr.exists():
            continue
        began = time.perf_counter()
        data = excerpt_wav(excerpt["meeting"], excerpt["start"], excerpt["end"])
        # load_and_preprocess_audio only calls .getvalue(), which BytesIO has.
        wav = app.load_and_preprocess_audio(cast(UploadedFile, io.BytesIO(data)))
        result = app.run_pipeline(wav, model, vad_model)
        asr.write_text(plain_text(result["transcript"]) + "\n")
        print(f"{excerpt['id']}: {time.perf_counter() - began:.1f}s")


def main() -> None:
    excerpts = load_labels()["excerpts"]
    fetch({e["meeting"] for e in excerpts})
    transcribe(excerpts)


if __name__ == "__main__":
    main()
