"""A/B: CompressARC vs CompressARC + symbolic hints, same seed and step budget per task.

Run from the repo root:  python -m arcsolver.ab --tasks <ids...> --steps 500
Writes one JSON line per (task, arm) to --out so partial runs are still usable.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .hinted import COMPRESSARC, HintedCompressor
from .search import partial_search

os.chdir(COMPRESSARC)  # CompressARC modules load dataset/ relative to cwd
import arc_compressor  # noqa: E402
import preprocessing  # noqa: E402
import solution_selection  # noqa: E402
import train  # noqa: E402

torch.set_default_device(os.environ.get("DEV", "cpu"))
torch.set_num_threads(int(os.environ.get("THREADS", "1")))


def attempts(log, n_test):
    a1 = [[list(r) for r in log.solution_most_frequent[i]] for i in range(n_test)]
    a2 = [[list(r) for r in log.solution_second_most_frequent[i]] for i in range(n_test)]
    return a1, a2


def run_arm(problem, solution, name, hints, steps, checkpoints):
    torch.manual_seed(0)
    np.random.seed(0)
    task = preprocessing.Task(name, problem, solution)
    model = HintedCompressor(task, hints) if hints is not None else arc_compressor.ARCCompressor(task)
    opt = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
    log = solution_selection.Logger(task)
    curve, t0 = {}, time.time()
    for step in range(steps):
        train.take_step(task, model, opt, step, log)
        if step + 1 in checkpoints:
            a1, a2 = attempts(log, task.n_test)
            curve[step + 1] = sum(any(a[i] == solution[i] for a in (a1, a2))
                                  for i in range(task.n_test)) / task.n_test
    gates = [float(g) for g in getattr(model, "gates", [])]
    return curve, gates, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="agi2eval")
    ap.add_argument("--tasks", nargs="*", help="task ids; default = all with a non-identity hint")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th task starting at i")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs" / "ab.jsonl"))
    args = ap.parse_args()

    problems = json.load(open(f"dataset/arc-agi_{args.split}_challenges.json"))
    solutions = json.load(open(f"dataset/arc-agi_{args.split}_solutions.json"))
    A = lambda x: np.array(x, dtype=np.int8)  # noqa: E731

    names = args.tasks or list(problems)
    si, sn = map(int, args.shard.split("/"))
    names = names[si::sn]
    checkpoints = {c for c in (100, 200, 300, 500, 750, 1000, 1500, 2000) if c <= args.steps} | {args.steps}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if Path(args.out).exists():
        done = {(r["task"], r["arm"]) for r in map(json.loads, open(args.out))}

    for name in names:
        p = problems[name]
        hints = partial_search([A(e["input"]) for e in p["train"]], [A(e["output"]) for e in p["train"]],
                               [A(e["input"]) for e in p["test"]], k=args.k)
        if not args.tasks and all(h.program == [] for h in hints):
            continue  # only identity (or nothing): the A/B would be uninformative
        for arm, arm_hints in (("base", None), ("hint", hints)):
            if (name, arm) in done:
                continue
            curve, gates, sec = run_arm(p, solutions[name], name, arm_hints, args.steps, checkpoints)
            rec = {"task": name, "arm": arm, "curve": curve, "gates": gates, "seconds": round(sec, 1),
                   "hints": [{"program": h.program, "train_acc": round(h.train_acc, 3)} for h in hints]}
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{name} {arm:4s} final {curve[args.steps]:.2f} gates {np.round(gates, 2).tolist()} "
                  f"{sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()
