"""Figures for the paper, drawn from results/ only.

    python scripts/make_figures.py      # -> paper/fig_*.png

Palette: validated 3-slot categorical (blue, orange, aqua) on a light surface. Aqua is below 3:1
contrast, so every series is also named in a direct label, never by colour alone.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R, OUT = ROOT / "results", ROOT / "paper"

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a897f", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True, "xtick.major.size": 0, "ytick.major.size": 0,
    "font.family": "DejaVu Sans",
})


def jl(name):
    return [json.loads(line) for line in open(R / name)]


def title(ax, main, sub):
    ax.set_title(main, loc="left", fontsize=13, fontweight="bold", color=INK, pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10, color=INK2, va="bottom")


def fig_failures():
    rows = json.load(open(R / "07_failure_analysis.json"))
    n = len(rows)
    order = [("correct", "Correct"), ("near_miss", "Near miss (≥90% cells right)"),
             ("far", "Right size & colours, <90% cells"), ("wrong_palette", "Wrong colours"),
             ("wrong_shape", "Wrong output size"), ("blank", "No answer")]
    cnt = collections.Counter(r["cat"] for r in rows)
    vals = [100 * cnt[k] / n for k, _ in order]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(order))[::-1]
    ax.barh(y, vals, height=0.55, color=[BLUE] + [MUTED] * (len(order) - 1))
    for yi, v, (k, _) in zip(y, vals, order):
        ax.text(v + 0.6, yi, f"{v:.0f}%  ({cnt[k]})", va="center", fontsize=10, color=INK2)
    ax.set_yticks(y, [lab for _, lab in order])
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel("share of test outputs (%)")
    ax.grid(axis="y", visible=False)
    title(ax, "Where CompressARC goes wrong",
          f"{n} ARC-AGI-1 evaluation test outputs, ~570 training steps each")
    fig.subplots_adjust(left=0.34, right=0.97, top=0.84, bottom=0.13)
    fig.savefig(OUT / "fig_failures.png", dpi=200)
    plt.close(fig)


def fig_round2():
    by = collections.defaultdict(dict)
    for r in jl("04_hint_ab_round2.jsonl"):
        by[r["task"]][r["arm"]] = r
    arms = [("base", "CompressARC, seed 0", BLUE, "-"), ("hint_noid", "+ program hints, seed 0", ORANGE, "-"),
            ("base_s1", "CompressARC, seed 1", AQUA, (0, (4, 3)))]
    steps = sorted(int(s) for s in next(iter(by.values()))["base"]["curve"])
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ends = []
    for arm, label, col, ls in arms:
        ys = [sum(v[arm]["curve"][str(s)] for v in by.values()) for s in steps]
        ax.plot(steps, ys, color=col, lw=2.5 if ls != "-" else 2, linestyle=ls,
                solid_capstyle="round", solid_joinstyle="round", dash_capstyle="round", zorder=3 if ls != "-" else 2)
        ax.plot(steps[-1], ys[-1], "o" if ls == "-" else "D", color=col, ms=8, mec=SURFACE, mew=2, zorder=4)
        ends.append((ys[-1], label + ("  (dashed)" if ls != "-" else "")))
    # direct labels at line ends, nudged apart when values tie
    placed = []
    for v, label in sorted(ends):
        yy = v
        while any(abs(yy - p) < 0.45 for p in placed):
            yy += 0.45
        placed.append(yy)
        ax.annotate(label, (steps[-1], v), xytext=(steps[-1] + 8, yy), va="center", fontsize=10,
                    color=INK2, annotation_clip=False)
    ax.set_xlim(steps[0] - 5, steps[-1] + 5)
    ax.set_ylim(0, max(e[0] for e in ends) + 1.5)
    ax.set_xlabel("training steps")
    ax.set_ylabel("tasks solved (of 40)")
    title(ax, "Program hints help no more than changing the random seed",
          "Round 2 (pre-registered): 40 unseen ARC-AGI-1 tasks. Seed 1 and hints coincide from step 200.")
    fig.subplots_adjust(left=0.09, right=0.70, top=0.84, bottom=0.13)
    fig.savefig(OUT / "fig_round2.png", dpi=200)
    plt.close(fig)


def fig_cover():
    """Cover: one ARC task, CompressARC's answer, a program hint, and the truth, side by side."""
    import sys
    sys.path.insert(0, str(ROOT))
    from arcsolver.search import partial_search

    ds = ROOT / "third_party" / "CompressARC" / "dataset"
    ch = json.load(open(ds / "arc-agi_evaluation_challenges.json"))
    sol = json.load(open(ds / "arc-agi_evaluation_solutions.json"))
    name = "4cd1b7b2"  # 4x4 Sudoku-like fill; small, readable, and has a real program hint
    t = ch[name]
    A = lambda x: np.array(x, dtype=np.int8)  # noqa: E731
    hints = partial_search([A(e["input"]) for e in t["train"]], [A(e["output"]) for e in t["train"]],
                           [A(e["input"]) for e in t["test"]])
    hint = next(h for h in hints if h.program)
    arc = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA",
           "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]  # standard ARC colours for grid cells
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(arc)
    panels = [(A(t["test"][0]["input"]), "test input"),
              (hint.test_outputs[0], "program hint\n(" + " > ".join(hint.program) + ")"),
              (A(sol[name][0]), "answer")]
    fig = plt.figure(figsize=(12, 6.3))
    fig.text(0.05, 0.86, "What is a compression-based\nARC solver missing?", fontsize=26,
             fontweight="bold", color=INK, va="top")
    fig.text(0.05, 0.55, "Symbolic hints for CompressARC:\nwhat helps, what doesn't, and why",
             fontsize=14, color=INK2, va="top")
    fig.text(0.05, 0.12, "ARC Prize 2026 · Paper Track · ARC-AGI-2", fontsize=11, color=MUTED)
    for i, (g, lab) in enumerate(panels):
        ax = fig.add_axes([0.50 + i * 0.165, 0.32, 0.14, 0.14 * 12 / 6.3 * g.shape[0] / g.shape[1]])
        ax.imshow(g, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, g.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, g.shape[0]), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        ax.grid(which="major", visible=False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(lab, fontsize=10, color=INK2, pad=8)
    fig.savefig(OUT / "fig_cover.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_failures()
    fig_round2()
    fig_cover()
    print("wrote", sorted(p.name for p in OUT.glob("fig_*.png")))
