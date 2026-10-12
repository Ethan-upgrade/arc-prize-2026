"""Recompute every number quoted in the paper from the raw files in results/.

    python scripts/reproduce_tables.py

No GPU, no Kaggle, no dataset download needed except for the failure-analysis feature table,
which reads ARC-AGI-1 evaluation tasks from third_party/CompressARC/dataset/ if present.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "results"


def jl(name):
    return [json.loads(line) for line in open(R / name)]


def final(curve):
    return curve[max(curve, key=int)]


def section(title):
    print(f"\n== {title}")


def repro():
    section("4.1 Reproduction / 4.2 correctness (8 ARC-AGI-1 eval tasks, 400 steps, T4)")
    for name, label in (("01_repro_eager_t4.jsonl", "eager"), ("02_repro_cudagraph_t4.jsonl", "CUDA graph")):
        rs = jl(name)
        solved = sum(final(r["curve"]) >= 1 for r in rs)
        sps = np.median([r["sec_per_step"] if "sec_per_step" in r else r["seconds"] / 400 for r in rs])
        print(f"  {label:10s}: solved {solved}/{len(rs)}, median {sps:.2f} s/step per process")


def hint_rounds():
    section("4.3 Whole-answer hints")
    for name, arms in (("03_hint_ab_round1.jsonl", ["base", "hint"]),
                       ("04_hint_ab_round2.jsonl", ["base", "base_s1", "hint_noid"])):
        by = collections.defaultdict(dict)
        for r in jl(name):
            by[r["task"]][r["arm"]] = r
        full = {t: v for t, v in by.items() if all(a in v for a in arms)}
        totals = {a: sum(final(v[a]["curve"]) for v in full.values()) for a in arms}
        gates = [max(v[a]["gates"]) for v in full.values() for a in arms if v[a]["gates"]]
        print(f"  {name}: {len(full)} tasks | solved " + ", ".join(f"{a}={totals[a]:g}" for a in arms)
              + (f" | gate>0.5 in {sum(g > 0.5 for g in gates)}/{len(gates)} hinted runs" if gates else ""))
        if "base_s1" in arms:
            diff = lambda a, b: sum(final(v[a]["curve"]) != final(v[b]["curve"]) for v in full.values())  # noqa: E731
            print(f"    outcome differs: base vs base_s1 = {diff('base', 'base_s1')}, "
                  f"base vs hint_noid = {diff('base', 'hint_noid')}")


def size_hint():
    section("4.5 Size-only hint (paired)")
    rs = jl("05_size_hint_paired.jsonl")
    b = sum(sum(final(r["curve"])["base"]) for r in rs)
    s = sum(sum(final(r["curve"])["size"]) for r in rs)
    differ = sum(any(v["base"] != v["size"] for v in r["curve"].values()) for r in rs)
    print(f"  {len(rs)} tasks | solved base={b} size={s} | tasks differing at any checkpoint: {differ}")


def submission():
    section("Submission v2 run (Kaggle public test file, T4, 4 procs)")
    rs = jl("06_submission_v2_per_task.jsonl")
    steps = sorted(r["steps"] for r in rs if "steps" in r)
    print(f"  tasks {len(rs)} | errors {sum('error' in r for r in rs)} | "
          f"steps min/median/max {steps[0]}/{steps[len(steps) // 2]}/{steps[-1]}")


def failures():
    section("4.4 Failure analysis (ARC-AGI-1 eval outputs in the public test file)")
    rows = json.load(open(R / "07_failure_analysis.json"))
    n = len(rows)
    for cat, k in collections.Counter(r["cat"] for r in rows).most_common():
        print(f"  {cat:14s} {k:3d} ({100 * k / n:.1f}%)")
    shrink = [r for r in rows if r["shape"] == "smaller"]
    print(f"  shrinking tasks solved: {sum(r['cat'] == 'correct' for r in shrink)}/{len(shrink)}")


if __name__ == "__main__":
    repro()
    hint_rounds()
    size_hint()
    submission()
    failures()
