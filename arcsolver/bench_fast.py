"""Check GraphedTrainer against train.take_step: same losses (up to sampling noise) and speed.

    python -m arcsolver.bench_fast <task> <steps>
"""
import json
import os
import sys
import time

import numpy as np
import torch

from .hinted import COMPRESSARC

os.chdir(COMPRESSARC)
torch.set_default_device("cuda")
import arc_compressor  # noqa: E402
import preprocessing  # noqa: E402
import solution_selection  # noqa: E402
import train  # noqa: E402

from .fast import GraphedTrainer  # noqa: E402


def make(name, split="evaluation"):
    problems = json.load(open(f"dataset/arc-agi_{split}_challenges.json"))
    sols = json.load(open(f"dataset/arc-agi_{split}_solutions.json"))
    torch.manual_seed(0)
    task = preprocessing.Task(name, problems[name], sols[name])
    return task, arc_compressor.ARCCompressor(task), sols[name]


def run(kind, name, steps, log_every):
    task, model, sol = make(name)
    log = solution_selection.Logger(task)
    losses = []
    if kind == "eager":
        opt = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
    else:
        tr = GraphedTrainer(task, model)
    torch.cuda.synchronize()
    t0 = time.time()
    for s in range(steps):
        if kind == "eager":
            train.take_step(task, model, opt, s, log)
            losses.append(log.loss_curve[-1])
        else:
            tr.step(log if s % log_every == 0 else None)
            if s % log_every == 0:
                losses.append(log.loss_curve[-1])
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    truth = tuple(tuple(map(tuple, g)) for g in sol)
    solved = int(any(x == truth for x in (log.solution_most_frequent, log.solution_second_most_frequent)))
    return dt, losses, solved


if __name__ == "__main__":
    name, steps = sys.argv[1], int(sys.argv[2])
    kinds = sys.argv[3].split(",") if len(sys.argv) > 3 else ["eager", "graph"]
    for kind, every in [(k, 1) for k in kinds]:
        dt, losses, solved = run(kind, name, steps, every)
        tail = np.mean(losses[-max(1, len(losses) // 5):])
        print(f"{kind:5s} log_every={every:2d}: {dt*1000:7.1f} ms/step  loss_start {losses[0]:9.1f}  "
              f"loss_end {tail:9.1f}  solved {solved}", flush=True)
