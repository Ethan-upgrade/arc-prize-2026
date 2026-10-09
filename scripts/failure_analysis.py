"""Classify why CompressARC's answers were wrong, using the v2 submission run on Kaggle's public
test file (which overlaps ARC-AGI-1 evaluation). No GPU needed.

    python scripts/failure_analysis.py

Categories (first match wins, per test output):
  correct        - one of the two attempts is exactly right
  blank          - no real answer (1x1 placeholder)
  wrong_shape    - best attempt has a different height/width than the truth
  wrong_palette  - right shape, but uses a colour absent from the truth or misses one present in it
  near_miss      - right shape and palette, >=90% of cells right
  far            - right shape and palette, <90% of cells right
Also tags each task with cheap structural features to see which kinds of task fail most.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "submission_run"
DATA = ROOT / "third_party" / "CompressARC" / "dataset"


def classify(attempts: dict, truth: list[list[int]]) -> tuple[str, float]:
    t = np.array(truth)
    cands = [np.array(attempts[k]) for k in ("attempt_1", "attempt_2")]
    if any(c.shape == t.shape and (c == t).all() for c in cands):
        return "correct", 1.0
    real = [c for c in cands if c.shape != (1, 1)]
    if not real:
        return "blank", 0.0
    same = [c for c in real if c.shape == t.shape]
    if not same:
        return "wrong_shape", 0.0
    best = max(same, key=lambda c: (c == t).mean())
    acc = float((best == t).mean())
    if set(np.unique(best)) != set(np.unique(t)):
        return "wrong_palette", acc
    return ("near_miss" if acc >= 0.9 else "far"), acc


def features(task: dict) -> dict:
    pairs = task["train"]
    ins = [np.array(p["input"]) for p in pairs]
    outs = [np.array(p["output"]) for p in pairs]
    same_shape = all(i.shape == o.shape for i, o in zip(ins, outs))
    out_fixed = len({o.shape for o in outs}) == 1
    smaller = all(o.size < i.size for i, o in zip(ins, outs))
    larger = all(o.size > i.size for i, o in zip(ins, outs))
    new_colours = any(set(np.unique(o)) - set(np.unique(i)) for i, o in zip(ins, outs))
    changed = np.mean([(i != o).mean() for i, o in zip(ins, outs) if i.shape == o.shape]) if same_shape else None
    return {
        "shape": "same" if same_shape else "fixed_out" if out_fixed else "smaller" if smaller else "larger" if larger else "varies",
        "new_colours": bool(new_colours),
        "changed_frac": None if changed is None else round(float(changed), 3),
        "max_cells": max(g.size for g in ins + outs),
    }


def main():
    sub = json.load(open(RUN / "submission.json"))
    ch = json.load(open(DATA / "arc-agi_evaluation_challenges.json"))
    sol = json.load(open(DATA / "arc-agi_evaluation_solutions.json"))
    ca = {}
    for f in (RUN / "ca").glob("*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            ca[r["task"]] = r
    names = [t for t in sub if t in sol]

    rows = []
    for t in names:
        feat = features(ch[t])
        for i, truth in enumerate(sol[t]):
            if i >= len(sub[t]):
                continue
            cat, acc = classify(sub[t][i], truth)
            rows.append({"task": t, "i": i, "cat": cat, "acc": round(acc, 3), "steps": ca.get(t, {}).get("steps"), **feat})

    n = len(rows)
    print(f"{len(names)} ARC-AGI-1 eval tasks, {n} test outputs\n")
    print("Failure categories:")
    for cat, k in collections.Counter(r["cat"] for r in rows).most_common():
        print(f"  {cat:14s} {k:3d}  ({100 * k / n:4.1f}%)")

    print("\nSolve rate by task feature:")
    for key in ("shape", "new_colours"):
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r[key]].append(r["cat"] == "correct")
        for g, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            print(f"  {key}={str(g):10s} n={len(v):3d}  solved {100 * np.mean(v):4.1f}%")

    wrong_shape = [r for r in rows if r["cat"] == "wrong_shape"]
    print("\nwrong_shape by task shape type:", dict(collections.Counter(r["shape"] for r in wrong_shape)))

    steps_ok = [r["steps"] for r in rows if r["cat"] == "correct" and r["steps"]]
    steps_bad = [r["steps"] for r in rows if r["cat"] != "correct" and r["steps"]]
    print(f"\nmedian steps: solved {np.median(steps_ok):.0f}, unsolved {np.median(steps_bad):.0f}")
    big = [r for r in rows if r["max_cells"] > 400]
    print(f"tasks with a grid >400 cells: n={len(big)}, solved {100 * np.mean([r['cat'] == 'correct' for r in big]):.1f}%")

    out = ROOT / "runs" / "failure_analysis.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nper-output rows -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
