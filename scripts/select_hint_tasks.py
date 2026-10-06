"""Pick tasks for the hint A/B: those with a non-identity hint that is NOT already exact.

Exact hints (train_acc == 1.0) are excluded: there the DSL alone already solves the task,
so a gain from the hint would say nothing about the hint *channel*.

    python scripts/select_hint_tasks.py --split evaluation --out runs/hint_tasks_agi1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arcsolver.search import partial_search  # noqa: E402

DATASET = ROOT / "third_party" / "CompressARC" / "dataset"


def analyse(item):
    name, task = item
    A = lambda x: np.array(x, dtype=np.int8)  # noqa: E731
    hints = partial_search([A(e["input"]) for e in task["train"]], [A(e["output"]) for e in task["train"]],
                           [A(e["input"]) for e in task["test"]])
    cells = max(len(g) * len(g[0]) for e in task["train"] + task["test"]
                for g in [e["input"]] + ([e["output"]] if "output" in e else []))
    return {"task": name, "cells": cells,
            "hints": [{"program": h.program, "train_acc": round(h.train_acc, 3)} for h in hints]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="evaluation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    tasks = json.load(open(DATASET / f"arc-agi_{args.split}_challenges.json"))
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(analyse, tasks.items()))

    exact = [r for r in rows if r["hints"] and r["hints"][0]["train_acc"] == 1.0]
    useful = [r for r in rows if r not in exact and any(h["program"] for h in r["hints"])]
    useful.sort(key=lambda r: r["cells"])  # small grids first: more tasks finish per GPU-hour
    print(f"{len(rows)} tasks | exact by DSL alone: {len(exact)} | non-trivial partial hint: {len(useful)}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"split": args.split, "exact": [r["task"] for r in exact],
                                          "tasks": useful}, indent=1))


if __name__ == "__main__":
    main()
