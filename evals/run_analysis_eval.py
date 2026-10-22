"""Score the transcript analysis against the AMI labels, on real weights.

Run from the repo root, after ``uv run python -m evals.prepare_ami``::

    uv run python -m evals.run_analysis_eval                  # held-out half
    uv run python -m evals.run_analysis_eval --split tune     # while tuning a prompt

It scores the app's own analysis functions and prompt (``evals.analysis``),
so tune a prompt by editing it in ``streamlit_app.py``, on the tuning half.
As a module, not a path: ``python evals/run_analysis_eval.py`` puts ``evals/``
rather than the repo root on the import path. Never in CI: it loads the LLM. Each run prints a table and writes a dated JSON file under
``evals/results/`` (git-ignored), which also carries every summary and the
excerpt's reference actions for the by-hand action-item recall and rubric.
"""

import argparse
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import mlx.core as mx

import streamlit_app as app
from evals.ami import Meeting
from evals.analysis import MODELS, analyse
from evals.metrics import brier, quantiles, reliability, roc_auc
from evals.prepare_ami import EVALS_DIR, MANUAL_DIR, TRANSCRIPTS_DIR, load_labels

RESULTS_DIR = EVALS_DIR / "results"
SOURCES = ("human", "asr")
SPLITS = {"tune": ("tune",), "held": ("held",), "all": ("tune", "held")}


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.2f}"


def score(rows: list[dict], questions: dict[str, str]) -> dict:
    """Board scores for one set of excerpts and one transcript source."""
    labels = [r["labels"][q] for r in rows for q in questions]
    p_yes = [r["p_yes"][q] for r in rows for q in questions]
    mass = [r["yes_no_mass"][q] for r in rows for q in questions]
    per_question = {
        q: roc_auc([r["labels"][q] for r in rows], [r["p_yes"][q] for r in rows])
        for q in questions
    }
    scored = [v for v in per_question.values() if v is not None]
    # What the board shows: one recording's questions sorted by P(yes). Scored
    # per excerpt across its questions, then averaged over the excerpts that
    # have both a yes and a no.
    within = [
        auc
        for r in rows
        if (
            auc := roc_auc(
                [r["labels"][q] for q in questions], [r["p_yes"][q] for q in questions]
            )
        )
        is not None
    ]
    return {
        "n_labels": len(labels),
        "auc_pooled": roc_auc(labels, p_yes),
        "auc_macro": sum(scored) / len(scored) if scored else None,
        "auc_per_question": per_question,
        "auc_within_excerpt": sum(within) / len(within) if within else None,
        "brier": brier(labels, p_yes),
        "reliability": reliability(labels, p_yes),
        "yes_no_mass_quantiles": dict(
            zip(("min", "p10", "median", "p90"), quantiles(mass, (0, 0.1, 0.5, 0.9)))
        ),
        "summary_parse_rate": sum(r["summary"] is not None for r in rows) / len(rows),
    }


def report(name: str, scores: dict, questions: dict[str, str]) -> None:
    print(f"\n== {name}: {scores['n_labels']} labels")
    print(
        f"ROC-AUC pooled {_fmt(scores['auc_pooled'])}   macro {_fmt(scores['auc_macro'])}"
        f"   within-excerpt {_fmt(scores['auc_within_excerpt'])}"
        f"   Brier {scores['brier']:.3f}   summary JSON parsed "
        f"{scores['summary_parse_rate']:.0%}"
    )
    for q, auc in scores["auc_per_question"].items():
        print(f"  {_fmt(auc)}  {questions[q]}")
    print("  reliability: P(yes) bin -> n, mean P(yes), yes rate")
    for low, high, n, mean, rate in scores["reliability"]:
        print(f"    {low:.1f}-{high:.1f}  {n:3d}  {_fmt(mean)}  {_fmt(rate)}")
    m = scores["yes_no_mass_quantiles"]
    print(
        f"  Yes+No mass: min {m['min']:.3f}  p10 {m['p10']:.3f}  "
        f"median {m['median']:.3f}  p90 {m['p90']:.3f}"
    )


def reference_actions(excerpt: dict) -> list[str]:
    """The corpus's abstractive action items linked into the excerpt's window."""
    if excerpt["meeting"].startswith("EN"):
        return []  # EN2002 has no abstractive summaries
    meeting = Meeting(MANUAL_DIR, excerpt["meeting"])
    return [
        s.text
        for s in meeting.summary_sentences()
        if s.kind == "actions"
        and any(excerpt["start"] <= t < excerpt["end"] for t in s.times)
    ]


def rescore(path: Path, labels: dict) -> None:
    """Print a saved run's scores under the current labels (e.g. after a review)."""
    saved = json.loads(path.read_text())
    current = {e["id"]: e["labels"] for e in labels["excerpts"]}
    questions: dict[str, str] = labels["questions"]
    for split in SPLITS[saved["split"]]:
        for source in SOURCES:
            rows = [
                {**r, "labels": current[r["id"]]}
                for r in saved["rows"][source]
                if r["split"] == split
            ]
            report(
                f"{split}/{source} ({saved['model']}, {saved['prompt']})",
                score(rows, questions),
                questions,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=SPLITS, default="held")
    parser.add_argument("--model", choices=MODELS, default="3b")
    parser.add_argument(
        "--rescore",
        type=Path,
        metavar="RESULTS_JSON",
        help="re-score a saved run against the current labels.json, without the model",
    )
    args = parser.parse_args()

    labels = load_labels()
    questions: dict[str, str] = labels["questions"]
    if args.rescore:
        rescore(args.rescore, labels)
        return
    excerpts = [e for e in labels["excerpts"] if e["split"] in SPLITS[args.split]]
    # The A7 limit: Phase 0 measured the 8B within budget only under it.
    mx.set_cache_limit(1 << 30)
    began = time.perf_counter()
    model, tokenizer = app.load_llm(*MODELS[args.model])
    load_seconds = time.perf_counter() - began

    rows: dict[str, list[dict]] = {source: [] for source in SOURCES}
    for excerpt in excerpts:
        for source in SOURCES:
            path = TRANSCRIPTS_DIR / f"{excerpt['id']}.{source}.txt"
            result = analyse(model, tokenizer, path.read_text().strip(), questions)
            rows[source].append(
                {
                    "id": excerpt["id"],
                    "split": excerpt["split"],
                    "labels": excerpt["labels"],
                    "reference_actions": reference_actions(excerpt),
                    **asdict(result),
                }
            )
            print(
                f"{excerpt['id']:>13} {source:>5}: {result.prefix_tokens} tokens, prefill "
                f"{result.prefix_tokens / result.prefill_seconds:.0f} tok/s, summary "
                f"{result.summary_seconds:.1f}s, questions {result.questions_seconds:.1f}s"
            )

    scores: dict[str, dict] = {}
    for split in SPLITS[args.split]:
        for source in SOURCES:
            name = f"{split}/{source}"
            scores[name] = score(
                [r for r in rows[source] if r["split"] == split], questions
            )
            report(name, scores[name], questions)

    prefill_tokens = sum(r["prefix_tokens"] for rs in rows.values() for r in rs)
    prefill_seconds = sum(r["prefill_seconds"] for rs in rows.values() for r in rs)
    speed = {
        "load_seconds": load_seconds,
        "prefill_tokens_per_second": prefill_tokens / prefill_seconds,
        "mean_analysis_seconds": sum(
            r["prefill_seconds"] + r["summary_seconds"] + r["questions_seconds"]
            for rs in rows.values()
            for r in rs
        )
        / sum(len(rs) for rs in rows.values()),
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
    }
    print(
        f"\nload {speed['load_seconds']:.1f}s, prefill "
        f"{speed['prefill_tokens_per_second']:.0f} tok/s, mean analysis "
        f"{speed['mean_analysis_seconds']:.1f}s, MLX peak {speed['peak_memory_gb']:.2f} GB"
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"analysis_{args.model}_{args.split}_shipped_{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "model": MODELS[args.model][0],
                "revision": MODELS[args.model][1],
                "prompt": "shipped",
                "prompt_text": {
                    "system": app.ANALYSIS_SYSTEM_PROMPT,
                    "transcript": app.ANALYSIS_TRANSCRIPT_PROMPT,
                    "question": app.QUESTION_PROMPT,
                    "summary": app.SUMMARY_PROMPT,
                    "summary_retry": app.SUMMARY_RETRY_PROMPT,
                },
                "split": args.split,
                "machine": platform.platform(),
                "speed": speed,
                "scores": scores,
                "rows": rows,
            },
            indent=1,
        )
    )
    print(f"wrote {Path(out).relative_to(EVALS_DIR.parent)}")


if __name__ == "__main__":
    main()
