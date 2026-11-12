"""Plot a preset's accuracy against the long-range keys a checkpoint reads at inference.

    lsa native curve <records> --preset ruler-4k --baseline decay128k-dense

Each N-LSA checkpoint is one line, trained at one `top_k` and read at several, with a ring on the point it trained at.
The dense checkpoint is a flat line.
A point is the mean over the preset's tasks of each task's mean score, the paper's RULER column for `ruler-4k`.
The records come from a sweep of `lsa native evaluate --suite benchmark`, which files every point under its own label.
"""

import argparse
import re
from pathlib import Path

import numpy as np

from lsa.training.native.benchmark import records, tasks
from lsa.training.native.benchmark.run import default_label
from lsa.training.native.constants import Kind


# A released checkpoint's run name ends in the `top_k` it was decayed at, as `decay128k-lsa-2048`
TRAINED_K = re.compile(r"-(\d+)$")


def trained_k(label: str) -> int | None:
    """Return the `top_k` a checkpoint was trained at, read off its run name.

    Args:
        label: The run the checkpoint was trained as.

    Returns:
        The `top_k`, or None when the name does not end in one.
    """
    match = TRAINED_K.search(label)
    return int(match.group(1)) if match else None


def preset_score(folder: Path, names: list[str]) -> float:
    """Return a run's mean over tasks of each task's mean score.

    Args:
        folder: The run's directory.
        names: The tasks to average.

    Returns:
        The score, from 0 to 1.

    Raises:
        FileNotFoundError: The run holds no samples for one of the tasks.
    """
    means = []
    for name in names:
        rows = records.read_samples(folder, name)
        means.append(np.mean([row["metrics"]["score"] for row in rows.values()]))
    return float(np.mean(means))


def collect(
    records_root: str | Path, preset: str, baseline: str
) -> tuple[dict[str, dict[int, float]], float | None]:
    """Read every point of a preset the records hold.

    Args:
        records_root: The records root.
        preset: The preset, a directory beside each run's label.
        baseline: The label of the dense run, drawn as a flat line.

    Returns:
        Each N-LSA checkpoint's score by the `top_k` it was read at, keyed by the run it was trained as, and the
        baseline's score or None when the records hold no baseline.
    """
    names = tasks.expand(preset)
    lines: dict[str, dict[int, float]] = {}
    dense = None
    for folder in sorted(Path(records_root).glob(f"*/{preset}")):
        if not (folder / records.RESULTS).exists():
            continue
        config = records.read_config(folder)
        if folder.parent.name == baseline:
            dense = preset_score(folder, names)
        elif config["attention"]["kind"] == Kind.LSA:
            line = lines.setdefault(default_label(config["checkpoint"]), {})
            line[int(config["attention"]["top_k"])] = preset_score(folder, names)
    return lines, dense


def table(lines: dict[str, dict[int, float]], dense: float | None) -> str:
    """Render the points as a markdown table, one row per `top_k`.

    Args:
        lines: Each checkpoint's score by `top_k`.
        dense: The baseline's score, or None.

    Returns:
        Markdown.
    """
    ks = sorted({k for line in lines.values() for k in line})
    header = ["top_k", *lines, *(["dense"] if dense is not None else [])]
    rows = [
        [str(k), *(f"{line[k]:.3f}" if k in line else "-" for line in lines.values())]
        + ([f"{dense:.3f}"] if dense is not None else [])
        for k in ks
    ]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    return "\n".join(
        " | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in [header, *rows]
    )


def render(
    lines: dict[str, dict[int, float]], dense: float | None, preset: str, baseline: str, out: Path
) -> None:
    """Draw the lines and write them as an image.

    Args:
        lines: Each checkpoint's score by `top_k`, keyed by the run it was trained as.
        dense: The baseline's score, or None.
        preset: The preset, named in the title.
        baseline: The dense run's label, named in the legend.
        out: Where the image goes.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    figure, axes = pyplot.subplots(figsize=(7, 4.5))
    for label, line in sorted(lines.items()):
        ks = sorted(line)
        [drawn] = axes.plot(ks, [line[k] for k in ks], marker="o", label=label)
        matched = trained_k(label)
        if matched in line:
            axes.plot(
                [matched],
                [line[matched]],
                marker="o",
                markersize=12,
                markerfacecolor="none",
                markeredgecolor=drawn.get_color(),
                markeredgewidth=2,
                linestyle="none",
            )
    if dense is not None:
        axes.axhline(dense, color="black", linestyle="--", label=baseline)
    axes.set_xscale("log", base=2)
    axes.set_title(f"{preset} Accuracy vs. Inference top_k")
    axes.set_xlabel("Inference Long-Range Keys (top_k)")
    axes.set_ylabel("Mean Score Over Tasks")
    axes.grid(alpha=0.3)
    axes.legend(title="Trained As (Ring Marks Trained top_k)")
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150)


def main() -> None:
    """Print the points of a preset and write the plot."""
    parser = argparse.ArgumentParser(description="Plot accuracy against the top_k read at inference")
    parser.add_argument("records", help="the records root")
    parser.add_argument("--preset", default="ruler-4k", help="the preset to plot")
    parser.add_argument("--baseline", default="decay128k-dense", help="the label of the dense run")
    parser.add_argument(
        "--out", default=None, help="the image - top_k_<preset>.png in the records root by default"
    )
    args = parser.parse_args()
    lines, dense = collect(args.records, args.preset, args.baseline)
    if not lines:
        raise SystemExit(f"{args.records} holds no N-LSA records of {args.preset}")
    print(table(lines, dense))
    out = Path(args.out) if args.out else Path(args.records) / f"top_k_{args.preset}.png"
    render(lines, dense, args.preset, args.baseline, out)
    print(f"wrote {out}")
