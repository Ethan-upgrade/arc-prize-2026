"""Baseline submission to the ARC-AGI-2 Kaggle competition (DSL search only).

    python scripts/kaggle_submit.py push     # create/update the submission notebook and run it
    python scripts/kaggle_submit.py status
    python scripts/kaggle_submit.py submit <version> "<message>"   # uses 1 of the daily submissions

The notebook reads the competition's test challenges (replaced by the hidden set at scoring time),
so it must never assume task ids or counts.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_ab import BUILD, DATASET_SLUG, kg, user  # noqa: E402

COMP = "arc-prize-2026-arc-agi-2"
KERNEL_SLUG = "arc2026-submission"

RUNNER = r'''
import glob, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

code = os.path.dirname(os.path.dirname(sorted(glob.glob("/kaggle/input/**/arcsolver/search.py", recursive=True))[0]))
sys.path.insert(0, code)
from arcsolver.evaluate import solve_task

challenges = sorted(glob.glob("/kaggle/input/**/arc-agi_test_challenges.json", recursive=True))[0]
tasks = json.load(open(challenges))
print(f"{len(tasks)} tasks from {challenges}", flush=True)

def blank(task):
    return [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in task["test"]]

def work(item):
    tid, task = item
    try:
        attempts, progs = solve_task(task, time_limit=PER_TASK_SECONDS)
        assert len(attempts) == len(task["test"])
        return tid, attempts, progs
    except Exception as e:  # one bad task must not sink the whole submission
        return tid, blank(task), [f"error: {e!r}"]

PER_TASK_SECONDS = 20
t0 = time.time()
submission, solved = {}, 0
with ProcessPoolExecutor(os.cpu_count()) as ex:
    for tid, attempts, progs in ex.map(work, tasks.items()):
        submission[tid] = attempts
        if progs and not progs[0].startswith("error") and progs[0] != "identity":
            solved += 1
for tid, task in tasks.items():  # every task id must be present
    submission.setdefault(tid, blank(task))
json.dump(submission, open("/kaggle/working/submission.json", "w"))
print(f"wrote {len(submission)} tasks; {solved} with a verified program; {time.time()-t0:.0f}s")
'''


def build(u: str) -> Path:
    d = BUILD / "submission"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "run.py").write_text(RUNNER)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{u}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [f"{u}/{DATASET_SLUG}"], "competition_sources": [COMP],
        "kernel_sources": []}))
    return d


if __name__ == "__main__":
    u = user()
    cmd = sys.argv[1]
    if cmd == "push":
        print(kg("kernels", "push", "-p", str(build(u))).strip())
    elif cmd == "status":
        print(kg("kernels", "status", f"{u}/{KERNEL_SLUG}", check=False).strip())
    elif cmd == "submit":
        version, message = sys.argv[2], sys.argv[3]
        print(kg("competitions", "submit", COMP, "-k", f"{u}/{KERNEL_SLUG}", "-v", version,
                 "-f", "submission.json", "-m", message).strip())
