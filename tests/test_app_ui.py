"""End-to-end UI tests driving the real main() widget tree via AppTest.

These complement test_streamlit_app.py (which mocks `st` entirely and never
exercises the UI wiring). AppTest re-executes streamlit_app.py in a fresh
namespace on every .run(), so mocks must patch the SHARED upstream imports
(mlx_audio, av) rather than streamlit_app.* — patching streamlit_app
attributes does not cross AppTest's script-runner boundary, and clicking Run
without an upstream patch would fetch the real ~0.9 GB speech model and the
MLX VAD checkpoint from the Hub.

The Run tests are hermetic: the speech model is stubbed at its loader, and the
MLX VAD loader is made to fail so load_vad_model takes its documented fallback
onto the PyTorch build that ships inside the silero-vad wheel — real VAD spans
on the fixture, nothing downloaded.

Layout assertions go through `at.sidebar`, `at.main` and the two `at.columns`
rather than the whole-tree accessors: AppTest walks main before the sidebar
and the input column before the transcript column, so `at.caption[0]` would
keep resolving to the filename only for as long as nothing rendered a caption
above it. Saying where an element lives makes the test document the layout
instead of depending on iteration order.
"""

import io
import tomllib
import wave
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import pytest
import streamlit as st
from streamlit import config
from streamlit.proto.Block_pb2 import Block
from streamlit.proto.Common_pb2 import FileURLs

# UploadedFileRec has no public export (streamlit.typing carries UploadedFile
# alone), and building a real one is what lets the recorder be faked with the
# type the app actually receives rather than a duck.
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.testing.v1 import AppTest

from streamlit_app import (
    ACCEPTED_FORMATS_CAPTION,
    MODEL_ID,
    MODEL_REVISION,
    TRANSCRIPT_MAX_WIDTH_PX,
)

APP = Path(__file__).parent.parent / "streamlit_app.py"
AUDIO_DIR = Path(__file__).parent / "data" / "audio"
CONFIG = Path(__file__).parent.parent / ".streamlit" / "config.toml"


@pytest.fixture
def audio_bytes() -> bytes:
    return (AUDIO_DIR / "sample_10s.wav").read_bytes()


@pytest.fixture
def app_config() -> dict:
    """The parsed .streamlit/config.toml. Read as UTF-8 explicitly, which TOML
    mandates and Streamlit itself passes — `read_text()` would otherwise decode
    the file's comments with whatever the platform locale happens to be."""
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


def _app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=60)


@pytest.fixture(autouse=True)
def _clear_streamlit_caches() -> Iterator[None]:
    # AppTest does not reset Streamlit's process-global @st.cache_resource store
    # between instances, so a cached (mocked or real) model could leak across
    # tests — which would make the Run-path loader.assert_called() guard
    # order-dependent. Clear it before every test for isolation.
    st.cache_resource.clear()
    yield


@pytest.fixture
def speech_loader() -> Iterator[MagicMock]:
    """The upstream MLX speech loader, patched to hand back a fake whose
    generate() returns "hello world". spec= keeps the fake honest: the app
    drives the model through generate() alone, so any other attribute access
    would be a reach into internals the real class need not have."""
    fake_model = MagicMock(spec=["generate"])
    fake_model.generate.return_value.text = "hello world"
    with patch("mlx_audio.stt.utils.load_model", return_value=fake_model) as loader:
        yield loader


@contextmanager
def _header_duration(seconds: float) -> Generator[MagicMock]:
    """Patch the shared `av.open` so audio_duration_seconds reads `seconds`
    from the (fake) stream header — no fixture is half an hour long. The
    stream's duration is in its own time base, which is how real containers
    carry it, so the app's `duration * time_base` arithmetic is exercised
    rather than bypassed."""
    stream = MagicMock(
        spec=av.AudioStream,
        duration=round(seconds * 16000),
        time_base=Fraction(1, 16000),
    )
    container = MagicMock()
    # The app asks for streams.best("audio"), not streams.audio[0], and narrows
    # the result with isinstance — hence the spec.
    container.streams.best.return_value = stream
    with patch("av.open") as opener:
        opener.return_value.__enter__.return_value = container
        yield opener


_RECORDING_SAMPLES = 2 * 16000


def _recording_bytes() -> bytes:
    """A valid wav of a length the fixture does not share, so the sample count
    handed to generate() identifies which source the run actually used —
    a caption or a filename stem can be right while audio_file is wrong."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * _RECORDING_SAMPLES)
    return buf.getvalue()


def _fixture_samples() -> int:
    """The wav fixture's frame count straight from its header, so the Run
    tests can bound the VAD segment against the clip without going through
    the decoder under test."""
    with wave.open(str(AUDIO_DIR / "sample_10s.wav")) as w:
        return w.getnframes()


@contextmanager
def _offline_mlx_vad() -> Generator[MagicMock]:
    """Make the MLX VAD loader fail as it would with no cache and no network.

    load_vad_model then falls back to the PyTorch build bundled in the
    silero-vad wheel, whose 16 kHz weights are bit-exact with the MLX
    checkpoint — so the spans are the real ones, and nothing is fetched.
    """
    with patch(
        "mlx_audio.vad.utils.load_model", side_effect=OSError("offline")
    ) as loader:
        yield loader


def _upload_and_run(audio_bytes: bytes, *, use_segmentation: bool) -> AppTest:
    at = _app().run()
    at.file_uploader[0].set_value(("sample_10s.wav", audio_bytes, "audio/wav"))
    at.toggle(key="use_segmentation").set_value(use_segmentation)
    at.run()
    at.button(key="transcribe").click()
    at.run()
    return at


def _assert_single_result_card(at: AppTest) -> None:
    """Exactly one card: one subheader, one transcript, one download button.
    The download filename is hashed into the served media URL rather than
    carried on the element, so it is pinned by the unit-level card test; here
    the stem it derives from is checked through session state instead."""
    assert not at.exception
    assert [s.value for s in at.subheader] == ["Transcription"]
    assert len(at.text) == 1
    (download,) = at.download_button
    assert download.key == "dl_txt"
    assert download.help == "Download transcription"
    assert download.icon == ":material/download:"
    assert download.proto.ignore_rerun is True  # on_click="ignore"
    # The card has no st.success banner by design, so the toast is the only
    # completion signal — and the only thing that would notice it going quiet.
    (toast,) = at.toast
    assert toast.value == "Transcription complete"
    assert toast.icon == ":material/check_circle:"
    assert at.session_state["result_stem"] == "sample_10s"
    # The card is in the transcript slot — the width-capped container in the
    # right column — and the slot holds nothing else: the next-step hint is
    # gone, and the input column has the player's filename and no card parts.
    # Block accessors recurse, so the column's containers are the slot, the
    # bordered card inside it and the card's header row. Picked by property
    # rather than unpacked by count, so a Streamlit release that renders a
    # spinner or an emptied progress bar as a block fails a named assertion
    # instead of a ValueError on the unpack.
    assert len(at.info) == 0
    input_col, transcript = at.columns
    containers = transcript.container
    (slot,) = [
        c
        for c in containers
        if c.proto.width_config.pixel_width == TRANSCRIPT_MAX_WIDTH_PX
    ]
    (card,) = [c for c in containers if c.proto.flex_container.border]
    (header,) = [
        c
        for c in containers
        if c.proto.flex_container.direction == Block.FlexContainer.HORIZONTAL
    ]
    assert [s.value for s in header.subheader] == ["Transcription"]
    assert len(header.download_button) == 1
    assert len(card.text) == 1
    assert len(slot.caption) == 0
    assert [c.value for c in input_col.caption] == [
        ACCEPTED_FORMATS_CAPTION,
        "sample_10s.wav",
    ]
    assert len(input_col.subheader) == 0
    assert len(input_col.download_button) == 0


def test_default_state() -> None:
    """App renders without exception in the documented default widget state:
    one toggle on, Run disabled, no results — and none of the prompt-era
    controls (source language, task pills, keywords) anywhere in the tree."""
    at = _app().run()
    assert not at.exception
    assert at.title[0].value == "Granite Speech Intelligence"
    assert len(at.segmented_control) == 0
    assert len(at.pills) == 0
    assert len(at.multiselect) == 0
    assert {t.key for t in at.toggle} == {"use_segmentation"}
    assert all(t.value is True for t in at.toggle)
    # Run is disabled until audio is loaded.
    assert at.button(key="transcribe").disabled is True
    assert len(at.subheader) == 0
    assert "result" not in at.session_state


def test_page_config_is_wide() -> None:
    """layout="wide" is the premise of the two-column workspace. The sidebar
    state is left at its default ("auto"): it holds a setting and the model
    name, nothing a run depends on, so collapsing it in a narrow window
    loses nothing. AppTest does not keep the PageConfig message — it is not
    a delta — so the call on the shared streamlit module is observed."""
    with patch("streamlit.set_page_config") as page_config:
        at = _app().run()
    assert not at.exception
    page_config.assert_called_once_with(
        page_title="Granite Speech Intelligence",
        page_icon=":material/graphic_eq:",
        layout="wide",
    )


def test_sidebar_holds_settings_and_app_info_only() -> None:
    """The sidebar is the VAD faux-label and the model line, nothing else; the
    whole run flow — input tabs, Transcribe, the player and the transcript —
    is in main, where it stays visible when the sidebar is collapsed.

    Both are markdown and neither is a caption: the model line moved off
    st.caption because caption opacity put its link under 4.5:1, so the
    zero-caption assertion is the guard against one coming back."""
    at = _app().run()
    assert not at.exception
    assert {t.key for t in at.sidebar.toggle} == {"use_segmentation"}
    # Two markdown elements and no caption: the VAD faux-label and the model
    # line, which is st.markdown(":small[...]") so its link escapes caption
    # opacity (see the comment on the call). Both are pinned by value below.
    assert len(at.sidebar.markdown) == 2
    assert len(at.sidebar.caption) == 0
    assert len(at.sidebar.file_uploader) == 0
    # audio_input has no typed accessor on AppTest; get() filters by proto type.
    assert len(at.sidebar.get("audio_input")) == 0
    assert len(at.sidebar.button) == 0
    assert len(at.sidebar.title) == 0
    assert len(at.main.file_uploader) == 1
    assert len(at.main.get("audio_input")) == 1
    assert [b.label for b in at.main.button] == ["Transcribe"]
    assert len(at.main.toggle) == 0
    assert [t.value for t in at.main.title] == ["Granite Speech Intelligence"]


def test_empty_state_keeps_the_grid() -> None:
    """With nothing loaded the two-column grid is already in place — the
    uploader and the disabled Run in the input column, a one-line caption in
    the transcript slot — so nothing moves when a file lands. No alert box:
    the uploader itself is the call to action."""
    at = _app().run()
    assert not at.exception
    assert len(at.info) == 0
    input_col, transcript = at.columns
    assert len(input_col.file_uploader) == 1
    assert [b.disabled for b in input_col.button] == [True]
    assert len(input_col.get("audio")) == 0  # no player...
    # ...so no filename under it: only the formats line.
    assert [c.value for c in input_col.caption] == [ACCEPTED_FORMATS_CAPTION]
    (slot,) = transcript.container
    assert slot.proto.width_config.pixel_width == TRANSCRIPT_MAX_WIDTH_PX
    (hint,) = slot.caption
    assert "Upload or record" in hint.value


def test_accepted_formats_are_listed_under_the_dropzone() -> None:
    """The dropzone prints the extensions on one line clipped to its width,
    so at laptop widths the video containers never show; a caption directly
    under it wraps. Pinned by value as well as by constant: the constant is
    derived from SUPPORTED_FORMATS and VIDEO_FORMATS, and a derivation that
    dropped a format or put one under the wrong heading would still equal
    itself."""
    at = _app().run()
    assert not at.exception
    assert ACCEPTED_FORMATS_CAPTION == (
        "Audio: WAV, FLAC, M4A, MP3, OGG, AAC · Video: MP4, MOV, WEBM, MKV"
    )
    upload_tab, _ = at.columns[0].tabs
    assert [e.type for e in upload_tab.children.values()] == [
        "file_uploader",
        "caption",
    ]
    assert upload_tab.caption[0].value == ACCEPTED_FORMATS_CAPTION


def test_sidebar_model_line_names_the_loaded_model() -> None:
    """The sidebar model line links the model card of the model the app
    actually loads and says English; there is no description under the
    title any more (the title is the only text in main above the grid).

    It is st.markdown with the :small[] directive rather than st.caption:
    caption dims its whole subtree to 60%, which put the only link on the
    page under 4.5:1 in both modes. :small[] is caption size at full
    linkColor, so the wrapper is asserted here too — dropping it would
    restore the size but not the contrast."""
    at = _app().run()
    assert not at.exception
    (model_line,) = [m for m in at.sidebar.markdown if MODEL_ID in m.value]
    # The whole string, not startswith/endswith: ":small[Model:] [link](url)]"
    # satisfies both ends while leaving the link outside the directive, which
    # is precisely the mis-scoping that would put it back at caption size.
    # The "English only" tail is the one language cue on screen — the CTC
    # model transcribes anything, so a wrong-language clip fails silently
    # without it — and it has to sit inside the directive too.
    assert model_line.value == (
        f":small[Model: [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) · English only]"
    )
    # By tree shape, not by element type: a description put back under the
    # title as *any* element would pass a per-type count somewhere, and it
    # would sit outside both columns, so nothing else on the page sees it.
    title, grid = at.main.children.values()
    assert title.type == "title"
    assert len(grid.columns) == 2


def test_faux_labels_are_bold() -> None:
    """The VAD pseudo-label renders as bold markdown so it reads as a form
    label (the real widget label is collapsed). Keywords went with the prompt:
    CTC has nothing to bias. Toxicity went with the guardian: there is no
    second model to score the transcript."""
    at = _app().run()
    assert not at.exception
    assert "**VAD segmentation**" in {m.value for m in at.sidebar.markdown}
    # The absence guards stay whole-tree: a faux-label creeping back next
    # to the uploader in main is exactly what they exist to catch.
    everywhere = {m.value for m in at.markdown}
    assert "**Toxicity check**" not in everywhere
    assert "**Keywords**" not in everywhere


def test_run_button_gating(audio_bytes: bytes) -> None:
    """Run enables as soon as audio is present — there is no task selection
    to wait on any more — and the input column gains the player with the
    filename under it, while the transcript slot swaps its empty-state line
    for a next-step caption (a caption, not an alert: it sits where the card
    will appear)."""
    at = _app().run()
    at.file_uploader[0].set_value(("sample_10s.wav", audio_bytes, "audio/wav"))
    at.run()
    assert not at.exception
    assert at.button(key="transcribe").disabled is False
    assert len(at.sidebar.button) == 0
    assert len(at.info) == 0
    input_col, transcript = at.columns
    assert [b.label for b in input_col.button] == ["Transcribe"]
    # The player (st.audio has no typed accessor either; get() by proto type)
    # and the filename under it, both in the input column.
    assert len(input_col.get("audio")) == 1
    assert len(transcript.get("audio")) == 0
    assert [c.value for c in input_col.caption] == [
        ACCEPTED_FORMATS_CAPTION,
        "sample_10s.wav",
    ]
    # Before a run the slot is the only container in the transcript column,
    # and the cap is on it — TestRenderResultCard pins that the card itself
    # carries no width, so this is the only place the constant is observed.
    (slot,) = transcript.container
    assert slot.proto.width_config.pixel_width == TRANSCRIPT_MAX_WIDTH_PX
    (hint,) = slot.caption
    assert hint.value.startswith("Press Transcribe")


@pytest.mark.parametrize(("duration", "blocked"), [(1801.0, True), (1799.0, False)])
def test_vad_off_long_audio_gates_run(
    audio_bytes: bytes, duration: float, blocked: bool
) -> None:
    """With VAD off, audio over MAX_VAD_OFF_DURATION_S disables Run behind a
    warning; just under it, neither. The boundary is pinned at 30 minutes on
    purpose: the message's "4 GB" is what a single inference measured at
    exactly that length (4.5 GB of MLX memory, before the decoded audio the
    process also holds), so retuning the constant has to retune the text."""
    with _header_duration(duration):
        at = _app().run()
        at.file_uploader[0].set_value(("long.wav", audio_bytes, "audio/wav"))
        at.toggle(key="use_segmentation").set_value(False)
        at.run()
    assert not at.exception
    assert at.button(key="transcribe").disabled is blocked
    input_col, transcript = at.columns
    assert [c.value for c in input_col.caption] == [
        ACCEPTED_FORMATS_CAPTION,
        "long.wav",
    ]
    if blocked:
        # Beside the button it disables, not under the sidebar toggle that
        # clears it: the sidebar can be collapsed, the button cannot.
        (vad_warning,) = input_col.warning
        assert "longer than 30 minutes" in vad_warning.value
        assert "more than 4 GB" in vad_warning.value
        assert vad_warning.icon == ":material/warning:"
        assert len(at.sidebar.warning) == 0
        # No "press Transcribe" beside a disabled button: the slot is empty.
        assert len(transcript.caption) == 0
    else:
        assert len(at.warning) == 0
        assert len(transcript.caption) == 1


def test_vad_off_short_clip_shows_no_warning(audio_bytes: bytes) -> None:
    """VAD off on a 30 s clip renders no warning at all. The prompted model
    fired a translation-passthrough warning here (anything past 20 s); an
    English-only CTC model has nothing to pass through, so the hour-long
    memory gate is the only VAD-off warning left."""
    with _header_duration(30.0):
        at = _app().run()
        at.file_uploader[0].set_value(("clip.wav", audio_bytes, "audio/wav"))
        at.toggle(key="use_segmentation").set_value(False)
        at.run()
    assert not at.exception
    assert len(at.warning) == 0
    assert at.button(key="transcribe").disabled is False


def test_the_most_recently_touched_input_is_the_source(
    audio_bytes: bytes,
) -> None:
    """st.tabs keeps both tabs mounted, so the uploader and the recorder can
    hold a value at once. `uploaded or recorded` used to settle that in the
    upload's favour permanently: a recording made after an upload landed in
    the widget and did nothing, with no way out but clearing the upload.
    select_source makes the most recently changed widget the source, so each
    tab's own value wins when the user touches it, and clearing the active
    one falls back to the other."""
    rec = UploadedFileRec("rec-id", "recording.wav", "audio/wav", audio_bytes)
    # The shared upstream command, not streamlit_app.audio_input: patching the
    # module's own attribute does not cross AppTest's script-runner boundary.
    with patch(
        "streamlit.audio_input", return_value=UploadedFile(rec, FileURLs())
    ) as recorder:
        at = _app().run()
        assert [c.value for c in at.columns[0].caption] == [
            ACCEPTED_FORMATS_CAPTION,
            "Recorded audio",
        ]

        # An upload lands after the recording and takes over.
        at.file_uploader[0].set_value(("sample_10s.wav", audio_bytes, "audio/wav"))
        at.run()
        assert not at.exception
        assert [c.value for c in at.columns[0].caption] == [
            ACCEPTED_FORMATS_CAPTION,
            "sample_10s.wav",
        ]

        # A NEW recording then takes it back — the case the old `or` lost.
        # A fresh file_id is what makes it new; the widget is otherwise equal.
        later = UploadedFileRec("rec-id-2", "recording.wav", "audio/wav", audio_bytes)
        recorder.return_value = UploadedFile(later, FileURLs())
        at.run()
        assert not at.exception
        assert [c.value for c in at.columns[0].caption] == [
            ACCEPTED_FORMATS_CAPTION,
            "Recorded audio",
        ]

        # A plain rerun changes nothing, so the choice stands.
        at.run()
        assert [c.value for c in at.columns[0].caption] == [
            ACCEPTED_FORMATS_CAPTION,
            "Recorded audio",
        ]

        # Clearing the inactive upload leaves the recording alone; clearing
        # the active source falls back to whatever is left.
        at.file_uploader[0].set_value(("sample_10s.wav", audio_bytes, "audio/wav"))
        at.run()
        assert [c.value for c in at.columns[0].caption] == [
            ACCEPTED_FORMATS_CAPTION,
            "sample_10s.wav",
        ]
        at.file_uploader[0].set_value(None)
        at.run()
    assert not at.exception
    assert [c.value for c in at.columns[0].caption] == [
        ACCEPTED_FORMATS_CAPTION,
        "Recorded audio",
    ]


def test_the_recording_supplies_the_download_stem_when_it_is_the_source(
    audio_bytes: bytes, speech_loader: MagicMock
) -> None:
    """The stem follows the chosen source, not merely whether an upload
    exists: with both set and the recording active, `uploaded` is still
    truthy, and keying the stem on it would name the file after an upload the
    transcript did not come from."""
    clip = _recording_bytes()
    rec = UploadedFileRec("rec-id", "recording.wav", "audio/wav", clip)
    with (
        _offline_mlx_vad(),
        patch(
            "streamlit.audio_input", return_value=UploadedFile(rec, FileURLs())
        ) as recorder,
    ):
        at = _app().run()
        # Upload first, so the recording has to be the later touch.
        at.file_uploader[0].set_value(("sample_10s.wav", audio_bytes, "audio/wav"))
        at.toggle(key="use_segmentation").set_value(False)
        at.run()
        assert at.session_state["_source"][2] == "upload"

        later = UploadedFileRec("rec-id-2", "recording.wav", "audio/wav", clip)
        recorder.return_value = UploadedFile(later, FileURLs())
        at.run()
        assert at.session_state["_source"][2] == "record"
        at.button(key="transcribe").click().run()
    assert not at.exception
    # The upload is still mounted and still truthy — keying either the stem or
    # audio_file on it rather than on the chosen source is the bug this pins.
    assert at.file_uploader[0].value is not None
    assert at.session_state["result_stem"].startswith("recording_")
    # VAD is off, so generate() sees the whole clip: the sample count is which
    # file was transcribed, and the two differ in length by construction.
    generate = speech_loader.return_value.generate
    assert generate.call_args.kwargs["audio"].shape == (_RECORDING_SAMPLES,)
    assert _RECORDING_SAMPLES != _fixture_samples()


def test_duration_cache_is_single_slot(audio_bytes: bytes) -> None:
    """With VAD off, the duration cache is a single `_duration` slot holding the
    current upload's (file_id, duration) — a new upload overwrites it rather
    than accumulating an entry per file, so it can't grow unbounded. The key is
    file_id, not (name, size): Streamlit mints one per upload and keeps it
    across reruns, so a bare rerun is a hit and an identical re-upload a miss.
    Hits and misses are told apart by counting `av.open` calls — the slot's key
    alone cannot, since a recompute under the same file_id stores the same key.
    Nothing but audio_duration_seconds opens a container before Run."""
    real_open = av.open
    with patch("av.open", wraps=real_open) as opener:
        at = _app().run()
        at.toggle(key="use_segmentation").set_value(False)
        at.file_uploader[0].set_value(("a.wav", audio_bytes, "audio/wav"))
        at.run()
        assert not at.exception
        first_id, first_duration = at.session_state["_duration"]
        assert first_id == at.file_uploader[0].value.file_id
        assert first_duration == pytest.approx(10.0, abs=0.05)
        assert opener.call_count == 1

        at.run()
        assert not at.exception
        # A rerun without a new upload is a hit: same key, no second read.
        assert at.session_state["_duration"][0] == first_id
        assert opener.call_count == 1

        at.file_uploader[0].set_value(("a.wav", audio_bytes, "audio/wav"))
        at.run()
        assert not at.exception
        # Same name, same bytes, new upload: a miss, read again, and the slot
        # now holds the new upload's id — overwritten in place, not accumulated.
        assert opener.call_count == 2
        assert at.session_state["_duration"][0] != first_id
        assert at.session_state["_duration"][0] == at.file_uploader[0].value.file_id
    # One slot, not one per upload: a `_duration_<file_id>` key would show here.
    duration_slots = [k for k in at.session_state if k.startswith("_duration")]
    assert duration_slots == ["_duration"]


def test_run_with_vad_on_renders_a_timestamped_transcript(
    audio_bytes: bytes, speech_loader: MagicMock
) -> None:
    """VAD on: one card with a timestamped transcript and nothing else. The VAD
    path trims the fixture's 0.76 s digital-silence lead-in, so the one segment
    starts past zero (still formatting as 0:00) and is shorter than the clip."""
    with (
        _offline_mlx_vad() as vad_loader,
        pytest.warns(RuntimeWarning, match="using the PyTorch VAD"),
    ):
        at = _upload_and_run(audio_bytes, use_segmentation=True)
    _assert_single_result_card(at)
    # Guards: every model came from a patch, not the Hub.
    speech_loader.assert_called_once_with(
        MODEL_ID, revision=MODEL_REVISION, strict=True
    )
    vad_loader.assert_called_once()

    assert at.text[0].value == "[0:00 - 0:09] hello world"
    generate = speech_loader.return_value.generate
    assert generate.call_count == 1
    assert generate.call_args.kwargs.keys() == {"audio"}
    segment = generate.call_args.kwargs["audio"]
    full_clip = _fixture_samples()
    assert 0 < segment.shape[0] < full_clip

    # No status banner of any kind: the card is transcript and download only.
    assert len(at.success) == 0
    assert len(at.warning) == 0


def test_run_with_vad_off_transcribes_the_whole_clip(
    audio_bytes: bytes, speech_loader: MagicMock
) -> None:
    """VAD off: no VAD load, one inference over every sample of the clip, and
    one timestamped line spanning it."""
    with _offline_mlx_vad() as vad_loader:
        at = _upload_and_run(audio_bytes, use_segmentation=False)
    _assert_single_result_card(at)
    vad_loader.assert_not_called()
    speech_loader.assert_called_once_with(
        MODEL_ID, revision=MODEL_REVISION, strict=True
    )

    assert at.text[0].value.splitlines() == ["[0:00 - 0:09] hello world"]
    generate = speech_loader.return_value.generate
    assert generate.call_count == 1
    full_clip = _fixture_samples()
    assert generate.call_args.kwargs["audio"].shape == (full_clip,)


def test_changing_a_toggle_discards_the_result(
    audio_bytes: bytes, speech_loader: MagicMock
) -> None:
    """The toggle is part of the input key, so flipping it after a run drops
    the stale card instead of leaving it beside a setting it no longer
    reflects."""
    with _offline_mlx_vad():
        at = _upload_and_run(audio_bytes, use_segmentation=False)
        _assert_single_result_card(at)
        speech_loader.assert_called_once_with(
            MODEL_ID, revision=MODEL_REVISION, strict=True
        )
        at.toggle(key="use_segmentation").set_value(True)
        at.run()
    assert not at.exception
    assert len(at.subheader) == 0
    assert "result" not in at.session_state
    assert "result_stem" not in at.session_state


def test_reuploading_an_identical_file_discards_the_result(
    audio_bytes: bytes, speech_loader: MagicMock
) -> None:
    """The input key is the upload's file_id, so a replacement with the same
    name and byte count — a re-exported file, or a second recording of the
    same length — is a new input and drops the stale card. Keyed on (name,
    size) it was a hit, and the previous transcript stayed on screen."""
    with _offline_mlx_vad():
        at = _upload_and_run(audio_bytes, use_segmentation=False)
        _assert_single_result_card(at)
        speech_loader.assert_called_once_with(
            MODEL_ID, revision=MODEL_REVISION, strict=True
        )
        at.file_uploader[0].set_value(("sample_10s.wav", audio_bytes, "audio/wav"))
        at.run()
    assert not at.exception
    assert len(at.subheader) == 0
    assert "result" not in at.session_state
    assert "result_stem" not in at.session_state


def test_undecodable_upload_shows_a_readable_error(speech_loader: MagicMock) -> None:
    """An unreadable upload gets the one-line st.error, not a traceback — and
    it gets it before the speech model loads, which is the whole reason the
    decode runs first: on a cold cache the model is a ~0.95 GB fetch, and a
    file that was never going to decode should not wait behind it.

    The bar is emptied in a `finally`, so nothing is left half-filled beside
    the error; there is no toast and no card."""
    at = _app().run()
    at.file_uploader[0].set_value(("bad.wav", b"not audio", "audio/wav"))
    at.toggle(key="use_segmentation").set_value(False)
    at.run()
    at.button(key="transcribe").click().run()

    assert not at.exception  # the RuntimeError is handled, not surfaced raw
    (error,) = at.error
    assert error.value.startswith("Failed to load audio file")
    assert error.icon == ":material/error:"
    # Decode before the model: the loader was never reached.
    speech_loader.assert_not_called()
    assert len(at.get("progress")) == 0  # finally: progress.empty()
    assert len(at.toast) == 0
    assert len(at.subheader) == 0
    assert "result" not in at.session_state
    # The error lands in the transcript slot, where the card would have.
    _input_col, transcript = at.columns
    assert len(transcript.error) == 1


def test_a_failed_rerun_does_not_leave_the_previous_card(
    audio_bytes: bytes, speech_loader: MagicMock
) -> None:
    """A second Run over the same input keeps `_last_input_key`, so the first
    run's result survives into it. If that second run then raises, the result
    has to go: `st.error` paints for one frame only (alerts are not sticky),
    so a kept result means the next rerun quietly re-renders the old
    transcript with the error gone — reading as though the retry worked.

    The failure is injected on the model rather than on `run_pipeline`:
    AppTest re-executes the script in a fresh namespace, so patching a
    `streamlit_app` attribute does not reach the code under test."""
    with _offline_mlx_vad():
        at = _upload_and_run(audio_bytes, use_segmentation=False)
        _assert_single_result_card(at)

        # The way an MLX allocation failure on a long clip surfaces.
        generate = speech_loader.return_value.generate
        generate.side_effect = RuntimeError("[metal::malloc] out of memory")
        at.button(key="transcribe").click().run()
        assert not at.exception
        assert [e.value for e in at.error] == ["[metal::malloc] out of memory"]
        assert len(at.subheader) == 0
        assert "result" not in at.session_state
        assert "result_stem" not in at.session_state

        # And the next rerun must not bring it back.
        at.run()
    assert not at.exception
    assert len(at.subheader) == 0
    assert len(at.text) == 0


def test_config_theme_defines_both_palettes(app_config: dict) -> None:
    """A custom theme has to define both sub-palettes, or the settings menu
    loses System / Light / Dark. The frontend only needs one of them — a
    [theme] table with nothing under [theme.light] or [theme.dark] is what
    drops the appearance section and locks the app to one mode (verified
    against 1.61.1 and 1.64.0 with a table carrying only primaryColor), and
    either sub-table keeps the menu with the other mode derived from [theme]
    — but a mode that is only derived is a mode nobody reviewed, so both are
    demanded here. The shipped theme (IBM Carbon, see CLAUDE.md) carries
    both; conditional so that dropping the theme altogether stays a valid
    state rather than a failing one."""
    if "theme" in app_config:
        # By value, not by header: an empty [theme.light] sets nothing, and the
        # frontend keeps the menu only when a sub-palette carries a value. It
        # looks for one recursively, so a sub-table holding nothing but an
        # empty [theme.light.sidebar] is empty to it too — a truthiness check
        # on the parsed dict would pass that and ship the collapsed menu.
        theme = app_config["theme"]
        assert _carries_a_value(theme.get("light"))
        assert _carries_a_value(theme.get("dark"))


def _carries_a_value(table: object) -> bool:
    """Whether a parsed TOML table sets at least one option, however nested:
    the frontend's own test for whether a sub-palette exists."""
    if isinstance(table, dict):
        return any(_carries_a_value(child) for child in table.values())
    return table is not None


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (None, False),
        ({}, False),
        ({"sidebar": {}}, False),  # an empty nested header sets nothing
        ({"primaryColor": "#0f62fe"}, True),
        ({"sidebar": {"backgroundColor": "#262626"}}, True),  # a nested value counts
    ],
)
def test_carries_a_value(table: object, expected: bool) -> None:
    assert _carries_a_value(table) is expected


def _option_paths(table: dict, valid: set[str]) -> Iterator[str]:
    """Flatten a parsed config.toml into the option paths it sets."""

    def walk(path: str, value: object) -> Iterator[str]:
        if path in valid:
            # A registered option, so stop rather than descend: a few options
            # take a table as their value (server.trustedUserHeaders) or a list
            # of them ([[theme.fontFaces]]), and those inner keys are free-form
            # rather than registry names.
            yield path
        elif isinstance(value, dict) and value:
            # An unregistered table with contents is a section, not a key.
            for key, child in value.items():
                yield from walk(f"{path}.{key}", child)
        else:
            # Scalar, list, or empty table under an unregistered path. The empty
            # case is what reports a typo'd section header with no keys of its
            # own, which would otherwise flatten to nothing and pass.
            yield path

    for key, value in table.items():
        yield from walk(key, value)


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ({"server": {"maxUploadSize": 500}}, ["server.maxUploadSize"]),
        # An option whose value is a table is a leaf; its keys are free-form.
        (
            {"server": {"trustedUserHeaders": {"X-Forwarded-User": "email"}}},
            ["server.trustedUserHeaders"],
        ),
        # As is one whose value is an array of tables.
        ({"theme": {"fontFaces": [{"family": "X", "url": "u"}]}}, ["theme.fontFaces"]),
        # Typos still surface at every depth and every shape.
        ({"theme": {"typoColor": "#fff"}}, ["theme.typoColor"]),
        ({"thheme": {"fontFaces": [{"family": "X"}]}}, ["thheme.fontFaces"]),
        ({"browserr": {}}, ["browserr"]),
    ],
)
def test_option_paths_stops_at_registered_options(
    table: dict, expected: list[str]
) -> None:
    """The shipped config exercises one path of _option_paths, so the rest are
    pinned here: descending into an option's own value fails a valid config,
    and not reaching a typo under an unregistered prefix passes an invalid one.
    """
    assert list(_option_paths(table, set(config._config_options_template))) == expected


def test_config_has_no_invalid_options(app_config: dict) -> None:
    """Every key in config.toml is a registered Streamlit config option.
    Unknown keys are logged and dropped rather than rejected, so a typo (or an
    option that only ever existed in a sibling table, as `base` does for
    [theme] but not [theme.light]) would otherwise go unnoticed."""
    valid = set(config._config_options_template)
    assert "theme.primaryColor" in valid  # registry is populated
    invalid = [path for path in _option_paths(app_config, valid) if path not in valid]
    assert invalid == [], f"invalid config options: {invalid}"


def test_config_serves_locally_without_usage_stats(app_config: dict) -> None:
    """The server stays on localhost and Streamlit's usage statistics stay off.
    Unset, server.address listens on every interface (verified 2026-10-17: it
    logged "started on :::8599" with a Network URL), so anyone on the same
    Wi-Fi could upload files for FFmpeg to parse and run the models; and
    gatherUsageStats defaults to on, in an app whose design is that nothing
    leaves the machine. test_config_has_no_invalid_options catches a typo'd
    key but not a deleted or changed one, which is what this pins — by value,
    with .get so a missing table reads as the assertion that failed."""
    assert app_config.get("server", {}).get("address") == "localhost"
    assert app_config.get("browser", {}).get("gatherUsageStats") is False
