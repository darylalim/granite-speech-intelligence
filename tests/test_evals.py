"""The evaluation's pure parts: its hand-written metrics and the committed split.

Nothing here loads a model or reads the corpus, which stays under the
git-ignored evals/data/ — so these run in CI where the evaluation itself never
does.
"""

import pytest

from evals.metrics import brier, quantiles, reliability, roc_auc
from evals.prepare_ami import load_labels, plain_text


@pytest.mark.parametrize(
    ("labels", "scores", "expected"),
    [
        ([True, False], [0.9, 0.1], 1.0),
        ([True, False], [0.1, 0.9], 0.0),
        ([True, False], [0.5, 0.5], 0.5),
        # 2 positives x 2 negatives: wins (0.8>0.3, 0.8>0.6, 0.4>0.3) = 3 of 4
        ([True, True, False, False], [0.8, 0.4, 0.3, 0.6], 0.75),
        ([True, True], [0.2, 0.9], None),
        ([False, False], [0.2, 0.9], None),
    ],
)
def test_roc_auc(
    labels: list[bool], scores: list[float], expected: float | None
) -> None:
    assert roc_auc(labels, scores) == expected


def test_brier() -> None:
    assert brier([True, False], [1.0, 0.0]) == 0.0
    assert brier([True, False], [0.5, 0.5]) == 0.25


def test_reliability_bins_include_one_and_keep_empty_rows() -> None:
    rows = reliability([True, False, True], [1.0, 0.05, 0.15], bins=5)
    assert len(rows) == 5
    assert rows[0] == (0.0, 0.2, 2, pytest.approx(0.1), 0.5)
    assert rows[1][2:] == (0, None, None)
    assert rows[4][2:] == (1, 1.0, 1.0)


def test_quantiles() -> None:
    assert quantiles([3.0, 1.0, 2.0], (0, 0.5, 1)) == [1.0, 2.0, 3.0]


def test_plain_text_strips_timestamps() -> None:
    transcript = "[0:00 - 0:09] hello there\n[0:10 - 0:20] general kenobi"
    assert plain_text(transcript) == "hello there general kenobi"


def test_labels_split_by_meeting_with_every_question_labelled() -> None:
    labels = load_labels()
    questions = set(labels["questions"])
    assert len(questions) == 9
    excerpts = labels["excerpts"]
    assert len(excerpts) == 20
    assert len({e["id"] for e in excerpts}) == 20
    halves: dict[str, set[str]] = {"tune": set(), "held": set()}
    for e in excerpts:
        assert set(e["labels"]) == questions
        assert all(isinstance(v, bool) for v in e["labels"].values())
        assert e["end"] - e["start"] == 180
        halves[e["split"]].add(e["meeting"])
    # Never one meeting's excerpts in both halves (PLAN.md, Evaluation).
    assert not halves["tune"] & halves["held"]
    assert sum(e["split"] == "tune" for e in excerpts) == 10


def test_labelled_questions_are_the_shipped_defaults() -> None:
    # The evaluation scores the board the app shows; if the defaults change,
    # the labels (and the recorded scores) no longer describe them.
    from streamlit_app import DEFAULT_QUESTIONS

    assert tuple(load_labels()["questions"].values()) == DEFAULT_QUESTIONS
