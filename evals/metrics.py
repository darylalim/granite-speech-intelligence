"""Scores for the yes/no board, by hand rather than via scikit-learn.

The evaluation may import only the standard library and what is already in
``uv.lock`` (docs/PLAN.md, Evaluation), and each of these is a few lines.
"""

from collections.abc import Sequence


def roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    """P(a random yes scores above a random no), ties counting half.

    The Mann-Whitney form of ROC-AUC, which is exactly the ranking promise the
    board makes. ``None`` when either class is missing: there is no ranking to
    score, and 0.5 would read as "no better than chance" rather than "unknown".
    """
    positives = [s for s, y in zip(scores, labels, strict=True) if y]
    negatives = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def brier(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Mean squared error of P(yes) against the 0/1 label."""
    pairs = list(zip(labels, scores, strict=True))
    return sum((s - y) ** 2 for y, s in pairs) / len(pairs)


def reliability(
    labels: Sequence[bool], scores: Sequence[float], bins: int = 5
) -> list[tuple[float, float, int, float | None, float | None]]:
    """Per equal-width bin of P(yes): ``(low, high, count, mean P(yes), yes rate)``.

    The last bin is closed on the right so a score of exactly 1.0 lands in it.
    Empty bins report ``None`` for both means rather than being dropped, so the
    table always has ``bins`` rows.
    """
    rows: list[tuple[float, float, int, float | None, float | None]] = []
    for i in range(bins):
        low, high = i / bins, (i + 1) / bins
        members = [
            (s, y)
            for s, y in zip(scores, labels, strict=True)
            if low <= s < high or (i == bins - 1 and s == high)
        ]
        if members:
            mean_score = sum(s for s, _ in members) / len(members)
            yes_rate = sum(y for _, y in members) / len(members)
            rows.append((low, high, len(members), mean_score, yes_rate))
        else:
            rows.append((low, high, 0, None, None))
    return rows


def quantiles(values: Sequence[float], points: Sequence[float]) -> list[float]:
    """Nearest-rank quantiles, for the Yes+No mass distribution."""
    ordered = sorted(values)
    return [
        ordered[min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))]
        for p in points
    ]
