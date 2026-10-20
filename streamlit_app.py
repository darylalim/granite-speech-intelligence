# Copyright 2026 Daryl Lim
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import math
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import av
import mlx.core as mx
import streamlit as st
import torch
import torchaudio
from mlx import nn
from mlx_audio.stt.utils import load_model as _load_stt_model
from mlx_audio.vad.utils import load_model as _load_mlx_vad_model
from silero_vad import get_speech_timestamps, load_silero_vad
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Encoder-only, English-only, greedy CTC: no prompt, no decoder, no
# translation. mlx_audio loads IBM's own bf16 checkpoint directly (no community
# conversion in between), which is why load_model can insist on strict=True.
MODEL_ID = "ibm-granite/granite-speech-5.0-470m-turboctc"
# Pinned even though the repo is org-maintained and loads strictly: strict only
# proves every weight key is present, not that the weights are the ones the
# constants below were measured against. A retrained checkpoint pushed to main
# would decode without complaint into different transcripts on the next cold
# cache. The hash is what makes a transcript reproducible.
MODEL_REVISION = "286456107c8ba1161f5c22dfe85466402c88333b"
# Silero VAD v6 in MLX form. Its 16 kHz weights are bit-exact with the
# `silero-vad` PyPI checkpoint the PyTorch fallback loads, so the two backends
# agree by construction rather than by luck — which is exactly what the pin
# below protects.
MLX_VAD_REPO = "mlx-community/silero-vad-v6"
# Pinned for a different reason than MODEL_REVISION, which guards against a
# personal account re-uploading. This repo is org-maintained, but mlx_audio's
# VAD loader takes `strict=False` and *cannot* do otherwise: the v6 conversion
# legitimately ships no `vad_8k.*` keys, so a strict load would reject it
# outright. Non-strict loading means renamed `vad_16k.*` keys would leave those
# layers randomly initialised — and the model would still build, still pass
# _supports_mlx_vad, still run clean, and return confident nonsense. That is
# the same silent-garbage mechanism documented for the 8 kHz branch in
# silero_vad(), and a content hash is the only thing that rules it out.
MLX_VAD_REVISION = "2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06"
SUPPORTED_FORMATS = [
    "wav",
    "flac",
    "m4a",
    "mp3",
    "ogg",
    "aac",
    "mp4",
    "mov",
    "webm",
    "mkv",
]
VIDEO_FORMATS = {"mp4", "mov", "webm", "mkv"}
# The dropzone prints the extensions on one line and clips it to its own width:
# on a 1470px laptop with the sidebar open it shows "WAV, FLAC, M4A, MP3, O…"
# and none of the video containers (measured 2026-10-17). A caption wraps
# instead, and is built from the lists so it cannot drift from what the
# uploader accepts.
ACCEPTED_FORMATS_CAPTION = (
    "Audio: "
    + ", ".join(f.upper() for f in SUPPORTED_FORMATS if f not in VIDEO_FORMATS)
    + " · Video: "
    + ", ".join(f.upper() for f in SUPPORTED_FORMATS if f in VIDEO_FORMATS)
)
SAMPLE_RATE = 16000
# Shortest clip generate() decodes. The encoder needs 4 stacked mel frames —
# 7 x 160-sample hops = 1120 samples at 16 kHz, 70 ms: two stride-2
# subsampling blocks halve the frame count twice, and the depthwise conv
# (kernel 7) needs a frame left. Anything shorter raises a ValueError from
# mlx_audio — a clear message under 10 ms, an stft length error or an opaque
# `[reshape]`/`[conv]` shape error above it (1119 samples fails, 1120
# decodes) — none of which main() turns into a clean st.error.
MIN_AUDIO_SAMPLES = 1120
# Silero's fixed framing at 16 kHz: a 512-sample (32 ms) decision chunk, each
# fed to the network together with the 64 samples preceding it.
VAD_CHUNK_SAMPLES = 512
VAD_CONTEXT_SAMPLES = 64
# Chunks per batched encoder pass. Caps peak memory rather than tuning speed:
# the STFT intermediate is (batch, 4, 258) float32, so 2048 chunks (~65s of
# audio) holds ~17 MB in flight. Measured +43 MB peak over the streaming path
# on 15 min of audio, and larger batches stopped helping once the GPU saturated.
VAD_ENCODER_BATCH_CHUNKS = 2048
# Ceiling for the VAD-off path, where the whole clip is one inference. Memory
# is the only bound: attention is block-local (128 frames per block, relative
# positions only within a block), so there is no context length to run out
# of, and the model transcribed an hour of continuous speech in one pass with
# no drift. Measured peak MLX memory (fresh process per run, bf16 weights =
# 0.88 GB resident): 10s -> 1.1 GB, 60s -> 1.7 GB, 300s -> 2.0 GB,
# 1200s -> 3.4 GB, 1800s -> 4.5 GB, 3600s -> 7.8 GB — linear at ~2 MB per
# second of audio. That series is MLX alone: the Streamlit process also
# holds the PyAV decode at the *source* rate and channel count (a 48 kHz
# stereo hour is ~1.4 GB float32 before the mono and 16 kHz copies), the
# upload bytes and the torch baseline, so an hour would land past 10 GB
# in-process. 30 minutes keeps the whole process around 7 GB on a 16 GB
# Mac; with VAD on there is no such ceiling at all. Both process figures
# include the since-removed guardian (125M fp32 parameters, ~500 MB; the
# transformers import itself stays, since mlx_audio's loader pulls it in),
# so they err high — the safe direction for a ceiling.
MAX_VAD_OFF_DURATION_S = 1800
# Target length for a segment fed to a single inference when VAD is on.
# Merging in get_speech_segments never exceeds it, but an unbroken VAD span
# (or the no-speech fallback) up to just under 2 x MIN_SEGMENT_DURATION_S
# (40s) is still fed whole — see the floor below. Unlike the LLM decoder
# this replaced, greedy CTC has no quality cliff with length — the cap exists
# so timestamp lines stay readable, not for accuracy. If anything splitting
# costs a little: on 73 LibriSpeech dev-clean utterances the WER was 6.2%
# whole vs 6.5% in 10s parts vs 8.2% in 5s parts, and on 64s of continuous
# speech 8.6% whole vs 9.1% at a 30s or 60s cap vs 10.1% at 15s. 30s is
# about three 128-frame attention blocks (10.24s each), so a forced part
# never sees less than a full block of context.
MAX_SEGMENT_DURATION_S = 30.0
# Floor for splitting an over-long span. Every forced split lands mid-utterance
# and costs a little accuracy (see the measurements above), so a span only
# modestly over the cap is left whole rather than halved: with the floor above
# half the cap, spans up to 2 x floor stay one part and only longer ones split.
MIN_SEGMENT_DURATION_S = 20.0
# Reading measure for the transcript. st.text wraps in the body font (16px,
# roughly 7.5px per character) inside a bordered card padded ~1em a side, so
# 720px is about 90 characters per line — the same measure as the centered
# layout this replaced. With the sidebar open on a 1440px laptop the column it
# sits in is narrower than this and the cap does nothing; past ~1600px of
# viewport the column outgrows it and an uncapped card would run well past 100
# characters a line. The extra width goes to the input column instead, where a
# larger video and a longer audio scrubber are worth having.
TRANSCRIPT_MAX_WIDTH_PX = 720


class PipelineResult(TypedDict):
    transcript: str


# (upload file_id, record file_id, "upload" | "record" | None) carried across
# reruns in st.session_state so select_source can see what changed.
type SourceState = tuple[str | None, str | None, str | None]


def is_video(filename: str) -> bool:
    return Path(filename).suffix.lower().lstrip(".") in VIDEO_FORMATS


def select_source(
    upload_id: str | None,
    record_id: str | None,
    previous: SourceState,
) -> SourceState:
    """Decide which input tab supplies the audio, most recent wins.

    Both tabs stay mounted, so the uploader and the recorder can hold a value
    at the same time and a plain `uploaded or recorded` silently keeps the
    upload — a recording made after one does nothing, with no way to say so.
    Whichever widget's `file_id` changed since the last rerun is the source;
    if neither changed the previous choice stands, and if the chosen one has
    since been cleared the other takes over. Returns the state to carry into
    the next rerun: `(upload_id, record_id, active)`.

    Pure, and keyed on ids rather than the files themselves, so the whole
    truth table is testable without a script run.
    """
    _prev_upload, _prev_record, active = previous
    if upload_id is not None and upload_id != _prev_upload:
        active = "upload"
    elif record_id is not None and record_id != _prev_record:
        active = "record"
    # The active source may have been cleared since it was chosen.
    if (active == "upload" and upload_id is None) or (
        active == "record" and record_id is None
    ):
        active = None
    if active is None:
        if upload_id is not None:
            active = "upload"
        elif record_id is not None:
            active = "record"
    return upload_id, record_id, active


def format_timestamp(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


_MLX_VAD_BRANCH_ATTRS = (
    "stft_conv",
    "conv1",
    "conv2",
    "conv3",
    "conv4",
    "lstm",
    "final_conv",
)
_MLX_VAD_BRANCH_CONFIG_ATTRS = ("pad", "cutoff", "chunk_size", "context_size")
_MLX_VAD_CONFIG_ATTRS = (
    "threshold",
    "min_speech_duration_ms",
    "min_silence_duration_ms",
    "speech_pad_ms",
)


def _supports_mlx_vad(model: Any) -> bool:
    """Probe the mlx_audio surface the batched VAD path reaches into.

    The fast path drives the branch submodules directly and borrows the
    private _probs_to_timestamps, so a rename upstream must degrade to the
    PyTorch reference rather than crash the run. This is the only place the
    app still reaches into mlx_audio internals — the speech model is driven
    through its public generate() alone.

    Beyond names this probe also checks two config *values*.
    The pin freezes the repo's config.json but not mlx_audio's BranchConfig
    defaults, which fill in any field the config omits — so a dependency
    upgrade can still move the framing out from under us. Neither this path nor
    mlx_audio's own _probs_to_timestamps (which hardcodes a 512-sample stride)
    could serve different framing correctly, and getting it wrong yields
    plausible garbage rather than an error, so refuse instead.
    """
    branch = getattr(model, "vad_16k", None)
    branch_config = getattr(branch, "config", None)
    config = getattr(model, "config", None)
    if branch is None or branch_config is None or config is None:
        return False
    if not (
        hasattr(model, "_probs_to_timestamps")
        and hasattr(model, "dtype")
        and all(hasattr(branch, name) for name in _MLX_VAD_BRANCH_ATTRS)
        and all(hasattr(branch_config, name) for name in _MLX_VAD_BRANCH_CONFIG_ATTRS)
        and all(hasattr(config, name) for name in _MLX_VAD_CONFIG_ATTRS)
    ):
        return False
    return (
        branch_config.chunk_size == VAD_CHUNK_SAMPLES
        and branch_config.context_size == VAD_CONTEXT_SAMPLES
    )


def _mlx_vad_windows(audio: mx.array) -> mx.array:
    """One (context + chunk) window per decision chunk, as a strided view.

    Frames the audio exactly as mlx_audio's streaming loop does — right-pad up
    to a chunk multiple, prepend the zero context — but the windows overlap by
    64 samples, so materialising them would copy ~12% more audio than it needs
    to. as_strided keeps them a view over the one padded buffer.
    """
    n_samples = audio.shape[-1]
    tail = (VAD_CHUNK_SAMPLES - n_samples % VAD_CHUNK_SAMPLES) % VAD_CHUNK_SAMPLES
    # Both edges in one pass: padding then concatenating the context would
    # allocate two full copies of the waveform, which on a 500 MB upload is a
    # lot of transient memory for a function that exists to avoid a copy.
    audio = mx.pad(audio, [(VAD_CONTEXT_SAMPLES, tail)])
    return mx.as_strided(
        audio,
        shape=(
            (n_samples + tail) // VAD_CHUNK_SAMPLES,
            VAD_CONTEXT_SAMPLES + VAD_CHUNK_SAMPLES,
        ),
        strides=(VAD_CHUNK_SAMPLES, 1),
    )


def _mlx_vad_probabilities(model: Any, audio: mx.array) -> mx.array:
    """Per-chunk speech probabilities, encoder batched across chunks.

    The STFT + conv encoder is stateless per chunk — it only ever sees that
    chunk's 576-sample window — so it can run on a whole batch at once. Only
    the LSTM is sequential, and nn.LSTM unrolls a full (1, T, 128) sequence
    inside one graph, so carrying (hidden, cell) across batches reproduces
    stepping chunk by chunk exactly. Net effect: one model call per 32 ms of
    audio becomes one per ~65s.

    Measured 2.9-3.5x faster than the PyTorch path across 10s-15min of audio,
    and identical where it counts: max |delta p| of 2.5e-06 over 9375 chunks,
    zero decisions flipped at the threshold, byte-identical spans out.

    mlx_audio's own get_speech_timestamps is *slower* than PyTorch here (0.66x),
    which is the whole reason this exists: at 309K parameters the per-call
    dispatch cost dominates the arithmetic, so call count is the only lever.
    """
    branch = model.vad_16k
    windows = _mlx_vad_windows(audio)
    pad, cutoff = branch.config.pad, branch.config.cutoff
    # PyTorch's ReflectionPad1d(right=pad), inlined: out[L + i] = in[L - 2 - i].
    reflect = mx.arange(windows.shape[-1] - 2, windows.shape[-1] - pad - 2, -1)

    outputs: list[mx.array] = []
    hidden = cell = None
    for start in range(0, windows.shape[0], VAD_ENCODER_BATCH_CHUNKS):
        batch = windows[start : start + VAD_ENCODER_BATCH_CHUNKS]
        x = mx.concatenate([batch, mx.take(batch, reflect, axis=-1)], axis=-1)
        x = branch.stft_conv(x[..., None])
        real, imag = x[..., :cutoff], x[..., cutoff:]
        x = mx.sqrt(real * real + imag * imag)
        x = nn.relu(branch.conv1(x))
        x = nn.relu(branch.conv2(x))
        x = nn.relu(branch.conv3(x))
        x = nn.relu(branch.conv4(x))
        if x.shape[1] != 1:
            # The batch axis is reused as the LSTM's sequence axis, which holds
            # only while the conv stack collapses each window to a single frame.
            # Different STFT framing leaves more (hop_length=64 gives 2), and
            # taking frame 0 would silently drop the rest while the reference
            # averages them. Refuse, and let the caller fall back to PyTorch.
            raise ValueError(
                f"VAD encoder produced {x.shape[1]} frames per chunk, expected 1"
            )
        hidden_seq, cell_seq = branch.lstm(x[:, 0, :][None], hidden=hidden, cell=cell)
        outputs.append(
            mx.squeeze(mx.sigmoid(branch.final_conv(nn.relu(hidden_seq))), axis=-1)[0]
        )
        hidden, cell = hidden_seq[:, -1, :], cell_seq[:, -1, :]
        mx.async_eval(outputs[-1], hidden, cell)
    return mx.concatenate(outputs) if outputs else mx.zeros((0,))


def _spans_in_seconds(
    speech_timestamps: list[dict[str, int]], sample_rate: int
) -> list[tuple[float, float]]:
    """Shared sample-offset -> seconds tail for both backends.

    The two paths are interchangeable only if they convert identically, so this
    is one function rather than two copies that could drift apart.
    """
    return [
        (ts["start"] / sample_rate, ts["end"] / sample_rate) for ts in speech_timestamps
    ]


def _mlx_vad_spans(
    wav: torch.Tensor, model: Any, sample_rate: int
) -> list[tuple[float, float]]:
    # Model.__call__ casts every input to model.dtype; this path bypasses it by
    # driving the branch directly, so do the cast here — MLX would otherwise
    # silently promote float32 audio against a float16 checkpoint and diverge
    # from the reference it claims to reproduce.
    audio = mx.array(wav.detach().reshape(-1).to(torch.float32).numpy()).astype(
        model.dtype
    )
    probabilities = _mlx_vad_probabilities(model, audio)
    mx.eval(probabilities)
    speech_timestamps = model._probs_to_timestamps(
        probabilities,
        audio_len=audio.shape[-1],
        sample_rate=sample_rate,
        threshold=model.config.threshold,
        min_speech_duration_ms=model.config.min_speech_duration_ms,
        min_silence_duration_ms=model.config.min_silence_duration_ms,
        speech_pad_ms=model.config.speech_pad_ms,
        return_seconds=False,
    )
    return _spans_in_seconds(speech_timestamps, sample_rate)


def _torch_vad_spans(
    wav: torch.Tensor, model: torch.nn.Module, sample_rate: int
) -> list[tuple[float, float]]:
    speech_timestamps = get_speech_timestamps(
        wav.squeeze(), model, sampling_rate=sample_rate
    )
    return _spans_in_seconds(speech_timestamps, sample_rate)


@st.cache_resource(show_spinner=False)
def _torch_vad_model() -> torch.nn.Module:
    """Cached PyTorch VAD, shared by every fallback site.

    load_silero_vad() is not memoised — it re-runs torch.jit.load on each call —
    so calling it inline would re-deserialise the model on every pipeline run of
    a session that permanently falls back, and leave a second copy resident
    alongside whatever load_vad_model already cached.
    """
    return load_silero_vad()


def silero_vad(
    wav: torch.Tensor, model: Any, sample_rate: int = SAMPLE_RATE
) -> list[tuple[float, float]]:
    if isinstance(model, torch.nn.Module):
        return _torch_vad_spans(wav, model, sample_rate)
    # The v6 conversion ships no 8 kHz weights, but mlx_audio builds that branch
    # regardless and loads non-strictly, so calling it at 8 kHz runs randomly
    # initialised layers and returns confident nonsense instead of failing.
    # Anything but 16 kHz goes to the PyTorch model, which carries both branches.
    if sample_rate != SAMPLE_RATE:
        return _torch_vad_spans(wav, _torch_vad_model(), sample_rate)
    try:
        return _mlx_vad_spans(wav, model, sample_rate)
    except Exception as e:  # noqa: BLE001 - any drift falls back to PyTorch
        # hasattr only catches renames. A changed signature or return shape gets
        # here instead, and would otherwise kill the run outright; warn so a
        # silent 3x slowdown does not become an undiagnosable mystery.
        warnings.warn(
            f"MLX VAD failed ({type(e).__name__}: {e}); falling back to the "
            "slower PyTorch model. mlx_audio's VAD internals have probably "
            "changed.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _torch_vad_spans(wav, _torch_vad_model(), sample_rate)


def _split_long_segment(
    start: float, end: float, max_duration: float
) -> list[dict[str, float]]:
    """Split an over-long span into equal parts.

    A span at or under max_duration is returned whole. Over it, the part
    count is the smallest that keeps every part at or above the floor
    (min(MIN_SEGMENT_DURATION_S, max_duration)), so a span under 2 x floor
    is also returned whole even though it exceeds max_duration — every
    forced split lands mid-utterance and costs accuracy.

    Equal parts rather than max_duration-sized chunks plus a remainder: at
    the shipped 30s cap and 20s floor, 30.1s stays whole, 45s becomes
    2 x 22.5s (not 30s + 15s), and 61s becomes 3 x ~20.3s.
    """
    duration = end - start
    if duration <= max_duration:
        return [{"start": start, "end": end}]
    parts = math.ceil(duration / max_duration)
    # Back off the part count rather than emit parts below the accuracy floor:
    # buffering pushes a natural 30.0s span to 30.6s, which would otherwise halve
    # into 2 x 15.3s. The floor cannot exceed the cap itself, or a caller passing
    # a small max_duration would stop getting any split at all.
    floor = min(MIN_SEGMENT_DURATION_S, max_duration)
    while parts > 1 and duration / parts < floor:
        parts -= 1
    step = duration / parts
    return [
        {"start": start + i * step, "end": start + (i + 1) * step} for i in range(parts)
    ]


def get_speech_segments(
    wav: torch.Tensor,
    model: Any,
    sample_rate: int = SAMPLE_RATE,
    max_segment_duration: float = MAX_SEGMENT_DURATION_S,
) -> list[dict[str, float]]:
    duration = wav.shape[-1] / sample_rate
    vad_segments = silero_vad(wav, model, sample_rate)
    if not vad_segments:
        return _split_long_segment(0.0, duration, max_segment_duration)
    start_buffer = 0.3
    end_buffer = 0.3
    min_gap = 0.5
    segments: list[dict[str, float]] = []
    for start, end in vad_segments:
        buffered_start = max(0.0, start - start_buffer)
        buffered_end = min(duration, end + end_buffer)
        # Merge across short gaps, but never past the cap — continuous speech
        # would otherwise collapse into one unbounded segment.
        if (
            segments
            and buffered_start - segments[-1]["end"] < min_gap
            and max(segments[-1]["end"], buffered_end) - segments[-1]["start"]
            <= max_segment_duration
        ):
            segments[-1]["end"] = max(segments[-1]["end"], buffered_end)
        else:
            # The buffers can pull this span's start behind the previous
            # segment's end. Merging used to absorb that overlap unconditionally
            # (a negative gap always satisfies the min_gap test), so the cap is
            # what first made this reachable: clamp instead, or the same audio
            # gets transcribed twice and timestamps run backwards.
            if segments:
                buffered_start = max(buffered_start, segments[-1]["end"])
                if buffered_start >= buffered_end:
                    # Wholly covered by the previous segment already.
                    continue
            segments.append({"start": buffered_start, "end": buffered_end})
    # A single VAD span can exceed the cap on its own (unbroken speech, or the
    # no-gap fallback above), so split whatever is still over.
    return [
        part
        for seg in segments
        for part in _split_long_segment(seg["start"], seg["end"], max_segment_duration)
    ]


@st.cache_resource(show_spinner=False)
def load_model(model_id: str, revision: str | None = None) -> Any:
    # strict=True: mlx_audio's default is strict=False, which leaves any weight
    # the checkpoint fails to supply randomly initialised — and greedy CTC
    # would still decode, into confident nonsense. That is the failure mode the
    # VAD loader documents and cannot guard against, because its checkpoint
    # legitimately omits keys. This one ships every key, so a missing one is a
    # renamed key upstream and an error worth raising at load time.
    return _load_stt_model(model_id, revision=revision, strict=True)


def _open_container(data: bytes) -> av.container.InputContainer:
    """Open an upload for reading, with tag text decoded lossily.

    `av.open` turns container *and* stream metadata into `str` at open time,
    before a frame is read, and its default is `errors="strict"`. FFmpeg's wav
    demuxer copies RIFF INFO and BWF `bext` text into those tags byte for
    byte (only ID3 is transcoded), and Windows/DAW tools write them in ANSI,
    so a perfectly decodable file raised UnicodeDecodeError on its artist
    field alone — and, because UnicodeDecodeError is a ValueError,
    `audio_duration_seconds` swallowed it into a silent None. Nothing here
    reads a tag, so "replace" costs nothing. One helper rather than a kwarg
    at each call site so the two opens cannot drift apart on it.
    """
    return av.open(io.BytesIO(data), mode="r", metadata_errors="replace")


def _best_audio_stream(
    container: av.container.InputContainer,
) -> av.AudioStream | None:
    """The track `av_find_best_stream` picks — the one flagged `default`,
    then the higher-bitrate / more-frames one — or None when the container
    has no audio.

    The same FFmpeg call the torchcodec path made, so a multi-track mp4/mkv
    keeps transcribing the track it always did. `streams.audio[0]` is the
    lowest-indexed track instead, which on a file whose commentary or
    low-bitrate track comes first is silently the wrong one, with the
    duration gate reading that track's length too. `best()` is typed
    `Stream | None`; the isinstance is what narrows it.
    """
    stream = container.streams.best("audio")
    return stream if isinstance(stream, av.AudioStream) else None


def _decode_audio(data: bytes) -> tuple[torch.Tensor, int]:
    """Decode a container's best audio stream to (channels, samples) float32
    at its native rate, via PyAV.

    PyAV is the decoder because its wheels *bundle* FFmpeg. torchaudio's own
    path goes through torchcodec, which dlopen()s whatever FFmpeg Homebrew
    has installed, and that broke twice in one week: Homebrew moved `ffmpeg`
    to 9.0 while torchcodec 0.15 only ships loaders for 4–8, and torchcodec
    0.16 added the 9 loader but dropped the /opt/homebrew rpath from its
    dylibs, so on Apple Silicon it finds nothing at all. Neither is fixable
    from inside the process (dyld reads its search paths at launch), so the
    app now depends on no system FFmpeg. torchaudio stays for its pure-torch
    resampler, which never touches torchcodec.

    Frames are converted to planar float32 at their own layout and
    rate before concatenation: a pure sample-format conversion, so swresample
    buffers nothing and every codec's output — packed or planar, s16 or
    fltp — lands in the same (channels, samples) matrix scaled to [-1, 1].
    """
    with _open_container(data) as container:
        stream = _best_audio_stream(container)
        if stream is None:
            raise ValueError("the file has no audio stream")
        # PyAV leaves codec_context None when the bundled FFmpeg has no
        # decoder for the track's codec (Dolby AC-4 in an mkv, say), and
        # AudioStream.__getattr__ then raises AttributeError for `layout` and
        # `rate` — "'AudioStream' object has no attribute 'layout'" is what
        # the user would see. Name the real cause before touching either.
        # `name` is proxied through that same __getattr__ on this build (the
        # shipped .py has a `Stream.name` property; the compiled extension
        # does not), hence the getattr rather than the attribute.
        if stream.codec_context is None:
            codec = getattr(stream, "name", None) or "unknown"
            raise ValueError(f"no decoder is available for the audio codec ({codec})")
        # The converter is built from the first decoded frame, not the stream
        # header. The two normally agree, but HE-AAC with implicit SBR is
        # the known exception: the header declares half the rate the frames
        # carry, and a converter targeting the header's rate would have
        # swresample downsample every frame to it — lossy, and reporting the
        # wrong rate to the resample that follows. The header only serves
        # the zero-frame case, where there is no frame to ask.
        rate, channels = stream.rate, len(stream.layout.channels)
        to_planar_float: av.AudioResampler | None = None
        frames: list[torch.Tensor] = []
        for frame in container.decode(stream):
            if to_planar_float is None:
                rate, channels = frame.sample_rate, len(frame.layout.channels)
                to_planar_float = av.AudioResampler(
                    format="fltp", layout=frame.layout, rate=rate
                )
            frames.extend(
                torch.from_numpy(converted.to_ndarray())
                for converted in to_planar_float.resample(frame)
            )
        if to_planar_float is not None:
            frames.extend(
                torch.from_numpy(converted.to_ndarray())
                for converted in to_planar_float.resample(None)
            )
    if not frames:
        return torch.zeros(channels, 0), rate
    return torch.cat(frames, dim=1), rate


def load_and_preprocess_audio(audio_file: UploadedFile) -> torch.Tensor:
    try:
        wav, sr = _decode_audio(audio_file.getvalue())
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {e}") from e

    # Caught here rather than downstream: an empty waveform survives VAD (which
    # falls back to a zero-length full-audio segment) and dies inside the
    # port's compute_features as a ValueError ("audio must contain at least
    # one 10 ms frame"), which main() surfaces as an st.exception traceback
    # where a RuntimeError gets the one-line st.error. Separate from the
    # length floor below because it has to run before the resample, which
    # raises its own opaque RuntimeError on an empty tensor at any rate but
    # 16 kHz.
    if wav.numel() == 0:
        raise RuntimeError("No audio detected: the file decoded to zero samples.")

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    # After the resample, so the floor is measured in the model's samples, not
    # the file's: 3000 samples at 48 kHz is 1000 here.
    if wav.shape[-1] < MIN_AUDIO_SAMPLES:
        raise RuntimeError(
            "Audio is too short: at least "
            f"{MIN_AUDIO_SAMPLES * 1000 // SAMPLE_RATE} ms is needed."
        )
    return wav


def audio_duration_seconds(audio_file: UploadedFile) -> float | None:
    """Clip length from the container headers alone — no frame is decoded.

    The stream's own duration (in its time base) is the accurate figure; the
    container-level one (in av.time_base microseconds) is the fallback for
    formats that carry only that. Neither is guaranteed, so None is a valid
    answer and the caller treats it as "unknown", not "zero".
    """
    try:
        with _open_container(audio_file.getvalue()) as container:
            stream = _best_audio_stream(container)
            if stream is None:
                return None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = container.duration / av.time_base
            else:
                return None
    except (av.FFmpegError, ValueError, OSError):
        return None
    if duration <= 0:
        return None
    return duration


@st.cache_resource(show_spinner=False)
def load_vad_model() -> Any:
    """Load the MLX Silero VAD v6 model, degrading to the PyTorch build.

    Two ways the fast path can be unavailable, and both land on the same
    fallback: the repo may not resolve at all (offline, or renamed), or it may
    load into a class whose internals have moved. Warn in either case — the
    PyTorch path is correct but ~3x slower, and a silent downgrade would be
    invisible until someone benchmarked it.
    """
    try:
        model = _load_mlx_vad_model(MLX_VAD_REPO, revision=MLX_VAD_REVISION)
    except Exception as e:  # noqa: BLE001 - any load failure falls back
        warnings.warn(
            f"Could not load {MLX_VAD_REPO} ({e}); using the PyTorch VAD.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _torch_vad_model()
    if not _supports_mlx_vad(model):
        warnings.warn(
            "mlx_audio's VAD internals have moved; using the PyTorch VAD.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _torch_vad_model()
    return model


def transcribe_audio(wav: torch.Tensor, model: Any) -> str:
    """One encoder pass and a greedy CTC collapse over a waveform slice.

    There is no prompt, no token budget and no autoregressive loop: generate()
    takes only the audio, and its cost is the single encoder pass — which is
    also why nothing is hoisted out of run_pipeline's segment loop any more.
    With the LLM decoder there were N prompts to amortise one encode across;
    here the encode *is* the whole call. Output is lowercase and unpunctuated,
    the way the model's training transcripts were normalised.
    """
    return model.generate(audio=wav.squeeze().numpy()).text


def run_pipeline(
    wav: torch.Tensor,
    model: Any,
    vad_model: Any = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    use_segmentation: bool = True,
) -> PipelineResult:
    if use_segmentation:
        assert vad_model is not None, "vad_model required when use_segmentation=True"
        segments = get_speech_segments(wav, vad_model)
    else:
        duration = wav.shape[-1] / SAMPLE_RATE
        segments = [{"start": 0.0, "end": duration}]

    lines: list[str] = []
    total_steps = max(len(segments), 1)
    for step, seg in enumerate(segments):
        if on_progress:
            on_progress(step, total_steps, f"segment {step + 1} of {total_steps}")
        # round(), not int(): the unsegmented path derives seg["end"] as
        # N / SAMPLE_RATE, and truncating that float round-trip drops the last
        # sample for ~0.6% of clip lengths (2002, 2006, ...).
        start_sample = round(seg["start"] * SAMPLE_RATE)
        end_sample = round(seg["end"] * SAMPLE_RATE)
        text = transcribe_audio(wav[:, start_sample:end_sample], model)
        ts_start = format_timestamp(seg["start"])
        ts_end = format_timestamp(seg["end"])
        lines.append(f"[{ts_start} - {ts_end}] {text}")

    # No closing (total, total) call: nothing runs after the last segment, and
    # main() empties the bar the moment this returns, so it would be coalesced
    # with that empty() in the same flush and, in practice, never painted.
    return {"transcript": "\n".join(lines)}


def _labeled_toggle(label: str, help: str, key: str, value: bool = True) -> bool:
    # A settings row: bold label left, switch right. A horizontal container
    # with a stretch space between them rather than st.columns([15, 1]): in
    # the sidebar (300px by default) a 1/16 column is ~15px, narrower than
    # the switch. The space is load-bearing — markdown's default width
    # resolves to "content" inside a horizontal container, so without it the
    # switch sits packed against the label instead of at the far edge.
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown(f"**{label}**", help=help)
        st.space("stretch")
        return st.toggle(label, value=value, label_visibility="collapsed", key=key)


def _render_result_card(result: PipelineResult, stem: str) -> None:
    transcript = result["transcript"]
    with st.container(border=True):
        # Download in the header row, not under the text: a long transcript
        # would otherwise put the only action a few hundred lines below the
        # fold. The stretch space floats it to the card's right edge.
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader("Transcription")
            st.space("stretch")
            st.download_button(
                # A real label, not "": download_button has no
                # label_visibility to collapse one, and with an empty label
                # the button's accessible name falls back to the Material
                # glyph's ligature text — "download", which is serviceable
                # only by luck of this icon's name (:material/file_download:
                # would announce "file_download"). help= is a hover tooltip
                # and contributes no name.
                "Download",
                transcript,
                f"{stem}_transcription.txt",
                "text/plain",
                key="dl_txt",
                icon=":material/download:",
                help="Download transcription",
                # Nothing reads the return value and the result already lives
                # in session state, so the default rerun does no work the app
                # wants — and it remounts the card, since the emptied progress
                # bar stops holding slot index 0 on the next run.
                on_click="ignore",
            )
        st.text(transcript)


def main() -> None:
    st.set_page_config(
        page_title="Granite Speech Intelligence",
        page_icon=":material/graphic_eq:",
        layout="wide",
    )

    # Settings and app info only; every control the run depends on is in main.
    with st.sidebar:
        use_segmentation = _labeled_toggle(
            "VAD segmentation",
            help=(
                "Splits audio into speech segments with timestamps using "
                "Silero VAD. Disable to process the whole audio in one pass."
            ),
            key="use_segmentation",
        )
        # "English only" is the one thing a user needs to know before uploading:
        # the model transcribes anything into confident lowercase English-shaped
        # text, so a wrong-language clip fails silently without the cue.
        #
        # st.markdown(":small[...]"), not st.caption: caption dims everything
        # inside it to 60% (the frontend sets opacity on the whole container),
        # which drags the model-card link to 3.3:1 dark / 2.5:1 light — under
        # the 4.5:1 this theme holds itself to, and the kept underline answers
        # "don't rely on colour alone", not contrast. The :small[] directive
        # sets font-size only, so the line keeps caption size at full
        # linkColor: 6.4:1 dark, 4.55:1 light on the sidebar. Not :gray[] for
        # the surrounding words — grayTextColor is a fixed built-in that lands
        # at 3.58:1 on the light sidebar and reintroduces the failure.
        st.markdown(
            f":small[Model: [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) "
            "· English only]"
        )

    st.title("Granite Speech Intelligence")

    # One grid in every state — input and source on the left, output on the
    # right — so nothing moves when a file lands or a transcript appears.
    input_col, transcript_col = st.columns([2, 3], gap="medium")
    with input_col:
        upload_tab, record_tab = st.tabs(["Upload", "Record"])
        with upload_tab:
            # No help= tooltip: the label is collapsed, so there is nothing for
            # it to sit beside, and a tooltip is where a keyboard or
            # screen-reader user may never reach. The accepted formats go in
            # the caption under the dropzone, which clips its own list.
            uploaded = st.file_uploader(
                "Upload an audio or video file",
                type=SUPPORTED_FORMATS,
                label_visibility="collapsed",
            )
            st.caption(ACCEPTED_FORMATS_CAPTION)
        with record_tab:
            recorded = st.audio_input("Record audio", label_visibility="collapsed")
        source = select_source(
            uploaded.file_id if uploaded else None,
            recorded.file_id if recorded else None,
            st.session_state.get("_source", (None, None, None)),
        )
        st.session_state["_source"] = source
        from_upload = source[2] == "upload"
        audio_file = uploaded if from_upload else recorded if source[2] else None

        if audio_file:
            if is_video(audio_file.name):
                st.video(audio_file)
            else:
                st.audio(audio_file)
            st.caption(audio_file.name if from_upload else "Recorded audio")

        vad_off_too_long = False
        if audio_file is not None and not use_segmentation:
            # Single-slot cache: av.open re-parses the container headers on
            # every rerun, so memoize the duration and recompute only when the
            # file changes. (getvalue() itself is O(1) — CPython hands back the
            # same bytes object for an unmodified BytesIO, the copy-on-write
            # UploadedFile.__init__ relies on; getbuffer() is the call that
            # would force the copy, so don't reach for it.) One slot can't
            # grow, so no eviction is needed. Keyed on file_id, which Streamlit
            # mints per upload (or recording) and keeps across reruns; (name,
            # size) took a same-length replacement for a hit and served the
            # stale duration.
            cached = st.session_state.get("_duration")
            if cached is None or cached[0] != audio_file.file_id:
                cached = (audio_file.file_id, audio_duration_seconds(audio_file))
                st.session_state["_duration"] = cached
            duration = cached[1]
            if duration is not None and duration > MAX_VAD_OFF_DURATION_S:
                vad_off_too_long = True
                # Next to the button it disables, not the toggle that clears
                # it: the toggle is in the sidebar, which can be collapsed.
                st.warning(
                    f"Enable VAD segmentation: audio is longer than "
                    f"{MAX_VAD_OFF_DURATION_S // 60} minutes. A single inference "
                    "that long needs more than 4 GB of memory on top of the "
                    "decoded audio.",
                    icon=":material/warning:",
                )

        can_run = audio_file is not None and not vad_off_too_long
        run_clicked = st.button(
            "Transcribe",
            key="transcribe",
            type="primary",
            icon=":material/transcribe:",
            disabled=not can_run,
            width="stretch",
        )

    # file_id rather than (name, size): a re-exported file or a second recording
    # of the same length would otherwise keep the previous transcript on screen.
    input_key = (audio_file.file_id, use_segmentation) if audio_file else None
    if input_key != st.session_state.get("_last_input_key"):
        for key in ("result", "result_stem"):
            st.session_state.pop(key, None)
        st.session_state["_last_input_key"] = input_key

    # Everything that answers "where is my transcript?" lands in this one slot:
    # the next-step hint, the progress bar and spinners during a run, an error,
    # and finally the card — so the eye never has to move.
    transcript_slot = transcript_col.container(width=TRANSCRIPT_MAX_WIDTH_PX)
    # Claimed in every state, not just on a run: the progress bar used to be
    # created inside the `if` below, so it held slot index 0 on the run's own
    # rerun and nothing held it afterwards — the card sat at index 1, then at
    # index 0 on the next rerun, and changing its delta path remounts it. An
    # emptied placeholder renders display:none, so reserving it costs no
    # layout and keeps the card at a fixed index the way the two-column grid
    # keeps everything else fixed.
    feedback = transcript_slot.empty()

    if run_clicked and can_run:
        assert audio_file is not None
        # Drop the previous run's result before starting, not after succeeding.
        # A re-run over the same input keeps `_last_input_key`, so the result
        # survives — and if this run then raises, `st.error` paints for exactly
        # one frame (alerts are not sticky) while the stale card comes back on
        # the next rerun, reading as though the retry had succeeded. Popping
        # up front makes failure leave nothing behind; success overwrites both
        # keys below, so the happy path is unchanged.
        for stale in ("result", "result_stem"):
            st.session_state.pop(stale, None)
        with transcript_slot:
            progress = feedback.progress(0, text="Starting transcription...")
            try:
                # Audio before the model: every decode-time RuntimeError
                # (unreadable file, zero samples, under the 70 ms floor) then
                # surfaces at once, instead of after a ~0.95 GB Hub fetch on a
                # cold cache. show_time on all three for one consistent
                # treatment, though only two of the waits can look hung —
                # minutes for a cold-cache fetch, seconds to decode a 500 MB
                # upload — and a static label gives no sign either is
                # advancing. The VAD load is 0.59s cold and 0.027s warm, at or
                # under st.spinner's own 0.5s delay, so its timer usually
                # never paints; it carries the kwarg so the three read alike.
                with st.spinner("Decoding audio...", show_time=True):
                    wav = load_and_preprocess_audio(audio_file)
                with st.spinner("Loading speech model...", show_time=True):
                    model = load_model(MODEL_ID, MODEL_REVISION)

                if use_segmentation:
                    with st.spinner("Loading VAD model...", show_time=True):
                        vad_model = load_vad_model()
                else:
                    vad_model = None

                def update_progress(i: int, total: int, label: str) -> None:
                    progress.progress(i / total, text=f"Transcribing {label}...")

                st.session_state.result = run_pipeline(
                    wav,
                    model,
                    vad_model,
                    on_progress=update_progress,
                    use_segmentation=use_segmentation,
                )
                if from_upload:
                    stem = Path(audio_file.name).stem
                else:
                    # Local wall-clock is what a user expects in a filename.
                    stem = datetime.now().strftime(  # noqa: DTZ005
                        "recording_%Y%m%d_%H%M%S"
                    )
                st.session_state.result_stem = stem
            except RuntimeError as e:
                st.error(str(e), icon=":material/error:")
                return
            except Exception as e:  # noqa: BLE001 - top-level UI error boundary
                st.exception(e)
                return
            finally:
                # Also on the error paths, or a half-filled bar labelled with
                # the segment that failed stays on screen next to the error.
                # Empties the placeholder, which stays in place holding the
                # position for the next run.
                feedback.empty()
        # The card carries no st.success banner by design, so this toast is
        # the only success signal; an icon is what makes it read as one.
        st.toast("Transcription complete", icon=":material/check_circle:")

    with transcript_slot:
        # Captions, not alerts: they sit where the card will appear. Nothing
        # while Run is blocked — the warning beside the button says why.
        if "result" in st.session_state:
            _render_result_card(st.session_state.result, st.session_state.result_stem)
        elif audio_file is None:
            st.caption("Upload or record audio to get started.")
        elif can_run:
            st.caption(
                "Press Transcribe. The transcript appears here, one timestamped "
                "line per segment."
            )


if __name__ == "__main__":
    main()
