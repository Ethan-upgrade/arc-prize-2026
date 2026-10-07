"""One submission worker: solve a list of tasks with CUDA-graph CompressARC under a wall-clock deadline.

    python -m arcsolver.submit_worker <challenges.json> <task_ids_csv> <out.jsonl> <deadline_epoch>

Each task gets an equal share of the time left (re-computed after every task), capped at
MAX_STEPS. One JSON line per task is flushed immediately, so a killed worker loses nothing done.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

from .hinted import COMPRESSARC

os.chdir(COMPRESSARC)
torch.set_default_device(os.environ.get("DEV", "cuda"))
import arc_compressor  # noqa: E402
import preprocessing  # noqa: E402
import solution_selection  # noqa: E402
import train  # noqa: E402

from .fast import GraphedTrainer  # noqa: E402

MAX_STEPS = 2000
MIN_SECONDS = 20
GRAPH_FAILED = 3


def solve(problem, name, seconds, graphed=True):
    """graphed=False is the plain CompressARC loop: slower, but no graph memory pool."""
    torch.manual_seed(0)
    task = preprocessing.Task(name, problem, None)
    model = arc_compressor.ARCCompressor(task)
    log = solution_selection.Logger(task)
    if graphed:
        trainer = GraphedTrainer(task, model)
    else:
        opt = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
    t_end = time.time() + seconds
    steps = 0
    while steps < MAX_STEPS and time.time() < t_end:
        if graphed:
            trainer.step(log)
        else:
            train.take_step(task, model, opt, steps, log)
        steps += 1
    attempts = [{"attempt_1": [list(r) for r in log.solution_most_frequent[i]],
                 "attempt_2": [list(r) for r in log.solution_second_most_frequent[i]]}
                for i in range(task.n_test)]
    return attempts, steps


def main():
    """Resumable: tasks already finished in out_path are skipped. If a CUDA graph fails, the task is
    recorded and the process exits with GRAPH_FAILED (a failed capture can leave CUDA unusable);
    the parent relaunches it and that task is retried in plain mode."""
    challenges_path, ids_csv, out_path, deadline = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    problems = json.load(open(challenges_path))
    names = [n for n in ids_csv.split(",") if n]
    finished, graph_failed = set(), set()
    if os.path.exists(out_path):
        for r in map(json.loads, open(out_path)):
            (graph_failed if "graph_error" in r and "attempts" not in r and "error" not in r else finished).add(r["task"])
    todo = [n for n in names if n not in finished]
    with open(out_path, "a") as out:
        for i, name in enumerate(todo):
            left = deadline - time.time()
            if left < MIN_SECONDS:
                break
            budget = max(MIN_SECONDS, left / (len(todo) - i))
            t0 = time.time()
            graphed = name not in graph_failed
            rec = {"task": name, "mode": "graph" if graphed else "eager"}
            try:
                rec["attempts"], rec["steps"] = solve(problems[name], name, budget, graphed=graphed)
            except Exception as e:
                if graphed:
                    rec["graph_error"] = f"{type(e).__name__}: {e}"[:200]
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(name, "graph failed:", rec["graph_error"], flush=True)
                    os._exit(GRAPH_FAILED)
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
                traceback.print_exc()
            rec["seconds"] = round(time.time() - t0, 1)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(name, rec["mode"], rec.get("steps"), rec["seconds"], rec.get("error", ""), flush=True)
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
