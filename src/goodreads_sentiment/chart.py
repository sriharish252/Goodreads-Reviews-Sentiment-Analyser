"""Scatter of star rating against review sentiment, one panel per model."""

from collections.abc import Sequence
from pathlib import Path
from statistics import fmean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from goodreads_sentiment.analysis import ModelResult  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
WARMER, COLDER, AGREE, BAND = "#2a78d6", "#e34948", "#898781", "#f0efec"
MAX_LABELS = 6


def save_chart(results: Sequence[ModelResult], path: Path) -> Path:
    fig, axes = plt.subplots(
        1, len(results), figsize=(6.4 * len(results), 5.2), squeeze=False, facecolor=SURFACE
    )
    for ax, result in zip(axes[0], results, strict=True):
        _panel(ax, result)
    fig.legend(
        handles=[
            _key("^", WARMER, "Reviews warmer than the stars"),
            _key("v", COLDER, "Reviews colder than the stars"),
            _key("o", AGREE, "Within 1 SD: ratings and reviews agree"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def _panel(ax, result: ModelResult) -> None:
    books = result.compared
    xs = [b.rating for b in books]
    ys = [b.sentiment for b in books]
    ax.set_facecolor(SURFACE)

    # Where z(sentiment) == z(rating), and the band |z(sentiment) - z(rating)| < threshold.
    mean_x, mean_y, sd_x, sd_y = fmean(xs), fmean(ys), pstdev(xs), pstdev(ys)
    lo, hi = min(xs) - 0.1, max(xs) + 0.1
    line = [mean_y + sd_y * (x - mean_x) / sd_x for x in (lo, hi)]
    band = result.threshold * sd_y
    ax.fill_between(
        (lo, hi), [y - band for y in line], [y + band for y in line], color=BAND, lw=0, zorder=0
    )
    ax.plot((lo, hi), line, color=MUTED, lw=1, zorder=1)

    for marker, color, group in (
        ("o", AGREE, [b for b in books if not b.flagged]),
        ("^", WARMER, [b for b in books if b.flagged and b.divergence > 0]),
        ("v", COLDER, [b for b in books if b.flagged and b.divergence < 0]),
    ):
        ax.scatter(
            [b.rating for b in group],
            [b.sentiment for b in group],
            marker=marker,
            s=(46 if len(books) <= 50 else 22) * (1 if marker == "o" else 1.4),
            color=color,
            edgecolors=SURFACE,
            linewidths=1,
            zorder=3,
        )

    for b in sorted(result.flagged, key=lambda b: -abs(b.divergence))[:MAX_LABELS]:
        title = b.title if len(b.title) <= 30 else b.title[:29] + "…"
        ax.annotate(
            title,
            (b.rating, b.sentiment),
            xytext=(6, 6 if b.divergence > 0 else -12),
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )

    share = f"{result.share_flagged:.0%}" if result.share_flagged is not None else "n/a"
    ax.set_title(
        f"{result.model}: {len(result.flagged)} of {len(books)} books diverge ({share}),"
        f" r = {result.correlation:.2f}",
        loc="left",
        fontsize=11,
        color=INK,
    )
    pad = 0.15 * (max(ys) - min(ys))
    ax.set_xlim(lo, hi)
    ax.set_ylim(max(-1.05, min(ys) - pad), min(1.05, max(ys) + pad))
    ax.set_xlabel("Average star rating", fontsize=9, color=INK_2)
    ax.set_ylabel("Average review sentiment (-1 to 1)", fontsize=9, color=INK_2)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")


def _key(marker: str, color: str, label: str) -> Line2D:
    return Line2D([], [], marker=marker, color=color, ls="", markersize=7, label=label)
