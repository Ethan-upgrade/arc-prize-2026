"""Solve tasks and score them with the official rule (2 attempts per test input, exact match)."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .search import search


def solve_task(task: dict, time_limit: float = 20.0) -> tuple[list[dict], list[str]]:
    tr_in = [np.array(p["input"], dtype=np.int8) for p in task["train"]]
    tr_out = [np.array(p["output"], dtype=np.int8) for p in task["train"]]
    te_in = [np.array(p["input"], dtype=np.int8) for p in task["test"]]
    res = search(tr_in, tr_out, te_in, time_limit=time_limit)
    attempts = []
    for i, g in enumerate(te_in):
        preds = [c.test_outputs[i].tolist() for c in res.candidates[:2]]
        while len(preds) < 2:
            preds.append(g.tolist())  # fallback: echo the input
        attempts.append({"attempt_1": preds[0], "attempt_2": preds[1]})
    progs = [" > ".join(c.program) or "identity" for c in res.candidates[:2]]
    return attempts, progs


def _work(args):
    tid, task, tl = args
    t0 = time.time()
    attempts, progs = solve_task(task, tl)
    return tid, attempts, progs, time.time() - t0


def score(task: dict, attempts: list[dict]) -> float:
    hits = [any(a[k] == t["output"] for k in ("attempt_1", "attempt_2"))
            for a, t in zip(attempts, task["test"])]
    return sum(hits) / len(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_repo/data/evaluation")
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="runs/latest.json")
    args = ap.parse_args()

    tasks = {p.stem: json.loads(p.read_text()) for p in sorted(Path(args.data).glob("*.json"))}
    jobs = [(tid, t, args.time_limit) for tid, t in tasks.items()]
    results, total = {}, 0.0
    with ProcessPoolExecutor(args.workers) as ex:
        for tid, attempts, progs, sec in ex.map(_work, jobs):
            s = score(tasks[tid], attempts)
            total += s
            results[tid] = {"score": s, "programs": progs, "seconds": round(sec, 2)}
            if s > 0:
                print(f"SOLVED {tid} {s:.2f} via {progs}")
    print(f"\nscore {total:.2f}/{len(tasks)} = {100 * total / len(tasks):.2f}%")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"score": total / len(tasks), "tasks": results}, indent=1))


if __name__ == "__main__":
    main()
