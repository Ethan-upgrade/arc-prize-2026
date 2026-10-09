"""Paired A/B for size hints: train CompressARC once per task, decode the same run two ways.

Size hints only change decoding, so both arms see identical training (same samples, same
weights); any difference is caused by the hint alone. No seed-noise control is needed.

    DEV=cuda python -m arcsolver.ab_size --split evaluation --steps 500 --out runs/ab_size.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from .hinted import COMPRESSARC

os.chdir(COMPRESSARC)
torch.set_default_device(os.environ.get("DEV", "cuda"))
import arc_compressor  # noqa: E402
import preprocessing  # noqa: E402
import solution_selection  # noqa: E402

from .fast import GraphedTrainer  # noqa: E402
from .size_hint import SizeHintLogger, predict_sizes  # noqa: E402


class Tee:
    """Feeds one training run's outputs to several loggers."""

    def __init__(self, *loggers):
        self.loggers = loggers

    def log(self, *args):
        for lg in self.loggers:
            lg.log(*args)


def solved(log, solution, n_test):
    a = (log.solution_most_frequent, log.solution_second_most_frequent)
    return [any([list(r) for r in x[i]] == solution[i] for x in a) for i in range(n_test)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="evaluation")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    problems = json.load(open(f"dataset/arc-agi_{args.split}_challenges.json"))
    solutions = json.load(open(f"dataset/arc-agi_{args.split}_solutions.json"))
    done = set()
    if Path(args.out).exists():
        done = {json.loads(l)["task"] for l in open(args.out)}

    for name, p in problems.items():
        hints = predict_sizes(p)
        if not hints or name in done:
            continue
        torch.manual_seed(0)
        task = preprocessing.Task(name, p, None)
        base = solution_selection.Logger(task)
        hinted = SizeHintLogger(task, hints)
        if not hinted.size_hints:
            continue  # every hint exceeds the canvas: nothing to compare
        model = arc_compressor.ARCCompressor(task)
        trainer = GraphedTrainer(task, model)
        tee = Tee(base, hinted)
        curve, t0 = {}, time.time()
        for step in range(args.steps):
            trainer.step(tee)
            if (step + 1) % 50 == 0:
                curve[step + 1] = {"base": solved(base, solutions[name], task.n_test),
                                   "size": solved(hinted, solutions[name], task.n_test)}
        rec = {"task": name, "hints": {str(k): v for k, v in hinted.size_hints.items()},
               "curve": curve, "seconds": round(time.time() - t0, 1)}
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        last = curve[args.steps]
        print(f"{name} base={last['base']} size={last['size']} {rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
