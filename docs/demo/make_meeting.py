"""Voice the demo meeting and build its WAV (docs/DEMO.md, Demo audio).

Usage, from the repo root (macOS only: the voices are `say`'s):

    PYTHONPATH=. uv run python docs/demo/make_meeting.py [--analyse]

Reads meeting_script.txt (one `speaker|line` per row), voices each line with
`say`, decodes it with the app's own _decode_audio and writes
docs/demo/build/team_meeting.wav: 16-bit 16 kHz mono, 0.5 s of lead-in and
1.2 s between lines. The 1.2 s is load-bearing: VAD merges any raw pause under
~1.1 s, and the transcript would come out as two arbitrary 30 s blocks instead
of one timestamped line per turn.

--analyse then runs the WAV through the app's transcription and analysis, as
the demo will show them, and prints the transcript, the summary and the board
with its replacement question scored too. Needs both models in the HF cache
(the 3B is ~3.9 GB on first use). PYTHONPATH=. is needed because a script run
by path gets its own folder on sys.path, not the repo root.
"""

import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

import streamlit_app as app

DEMO = Path(__file__).resolve().parent
SCRIPT = DEMO / "meeting_script.txt"
WAV = DEMO / "build" / "team_meeting.wav"
VOICES = {"A": "Daniel (English (UK))", "B": "Eddy (English (US))"}
LEAD_S = 0.5
GAP_S = 1.2
REPLACEMENT = "Is a bug or problem discussed?"  # storyboard shot 7


class Upload:
    """Just enough of an UploadedFile for load_and_preprocess_audio."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def build_wav() -> Path:
    rate = app.SAMPLE_RATE
    gap = np.zeros(int(GAP_S * rate), dtype=np.float32)
    parts = [np.zeros(int(LEAD_S * rate), dtype=np.float32)]
    rows = [r for r in SCRIPT.read_text().splitlines() if r.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        for i, row in enumerate(rows, 1):
            speaker, text = row.split("|", 1)
            aiff = Path(tmp) / f"line{i:02d}.aiff"
            subprocess.run(
                ["say", "-v", VOICES[speaker], "-o", str(aiff), text], check=True
            )
            parts += [app._decode_audio(aiff.read_bytes()), gap]
    pcm = np.clip(np.concatenate(parts), -1, 1)
    WAV.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(WAV), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((pcm * 32767).astype("<i2").tobytes())
    print(f"wrote {WAV} ({len(pcm) / rate:.1f}s, {len(rows)} lines)")
    return WAV


def analyse(path: Path) -> None:
    app.limit_mlx_cache()
    wav = app.load_and_preprocess_audio(Upload(path.read_bytes()))  # ty: ignore[invalid-argument-type]
    model = app.load_model(app.MODEL_ID, app.MODEL_REVISION)
    vad = app.load_vad_model()
    t = time.perf_counter()
    result = app.run_pipeline(wav, model, vad)
    print(f"\n--- transcript ({time.perf_counter() - t:.2f}s) ---")
    print(result["transcript"])

    llm, tok = app.load_llm(app.LLM_ID, app.LLM_REVISION)
    t = time.perf_counter()
    summary, board = app.run_analysis(
        llm,
        tok,
        app.segments_text(result["segments"]),
        [*app.DEFAULT_QUESTIONS, REPLACEMENT],
        summarise=True,
    )
    print(f"\n--- summary ({time.perf_counter() - t:.2f}s incl. board) ---")
    for key, value in (summary or {}).items():
        print(f"{key}: {value}")
    print("\n--- board (the last default is replaced in shot 7) ---")
    for question, p in sorted(board or [], key=lambda qp: -qp[1]):
        print(f"{p:.2f}  {question}")


def main() -> None:
    path = build_wav()
    if "--analyse" in sys.argv:
        analyse(path)


if __name__ == "__main__":
    main()
