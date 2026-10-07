"""Submission notebook for the ARC-AGI-2 Kaggle competition.

v1 (10-05): DSL search only. v2 (10-07): DSL exact programs + CUDA-graph CompressARC on GPU.

    python scripts/kaggle_submit.py push     # create/update the submission notebook and run it
    python scripts/kaggle_submit.py pull     # download the run's submission.json + logs
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

RUNNER_V2 = r'''
import glob, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor

T0 = time.time()
TOTAL_HOURS = 11.0          # Kaggle kills at 12h; keep an hour for setup, assembly and slack
N_PROCS = 4
code = os.path.dirname(os.path.dirname(sorted(glob.glob("/kaggle/input/**/arcsolver/submit_worker.py", recursive=True))[0]))
sys.path.insert(0, code)
# The competition's file, never a copy that happens to sit in a code dataset.
found = sorted(glob.glob("/kaggle/input/**/arc-agi_test_challenges.json", recursive=True))
comp = [f for f in found if "arc-prize-2026-arc-agi-2" in f]
assert comp, f"competition test file not found among {found}"
challenges = comp[0]
tasks = json.load(open(challenges))
print(f"{len(tasks)} tasks from {challenges}", flush=True)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout, "cpus", os.cpu_count(), flush=True)

def blank(task):
    return [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in task["test"]]

# 1) DSL search: keep only exact programs (verified on every train pair).
from arcsolver.evaluate import solve_task
def dsl(item):
    tid, task = item
    try:
        attempts, progs = solve_task(task, time_limit=8)
        return tid, (attempts if progs and progs[0] != "identity" and len(attempts) == len(task["test"]) else None)
    except Exception:
        return tid, None
with ProcessPoolExecutor(os.cpu_count()) as ex:
    dsl_out = dict(ex.map(dsl, tasks.items()))
print(f"DSL exact on {sum(v is not None for v in dsl_out.values())} tasks, {time.time()-T0:.0f}s", flush=True)

# 2) CompressARC, N_PROCS workers sharing the GPU. Large grids first so they get time early.
size = lambda t: max(len(g) * len(g[0]) for e in t["train"] + t["test"] for g in [e["input"]] + ([e["output"]] if "output" in e else []))
order = sorted(tasks, key=lambda k: -size(tasks[k]))
deadline = T0 + TOTAL_HOURS * 3600
os.makedirs("/kaggle/working/ca", exist_ok=True)
def launch(j):
    out = f"/kaggle/working/ca/w{j}"
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "arcsolver.submit_worker", challenges, ",".join(order[j::N_PROCS]),
         out + ".jsonl", str(deadline)],
        cwd=code, env={**os.environ, "DEV": "cuda", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        stdout=open(out + ".log", "a"), stderr=subprocess.STDOUT)
procs = {j: launch(j) for j in range(N_PROCS)}
restarts = 0
while procs and time.time() < deadline + 300:
    for j, p in list(procs.items()):
        code_ = p.poll()
        if code_ is None:
            continue
        if code_ == 3 and time.time() < deadline - 60 and restarts < 50:  # graph failed: resume in a fresh process
            procs[j] = launch(j)
            restarts += 1
        else:
            del procs[j]
    time.sleep(10)
for p in procs.values():
    p.kill()
print(f"worker restarts after graph failures: {restarts}", flush=True)

ca = {}
for f in glob.glob("/kaggle/working/ca/*.jsonl"):
    for line in open(f):
        r = json.loads(line)
        if "attempts" in r and len(r["attempts"]) == len(tasks[r["task"]]["test"]):
            ca[r["task"]] = r

# 3) Assemble: DSL exact answer first, CompressARC fills the remaining attempt(s).
submission = {}
for tid, task in tasks.items():
    out = []
    for i in range(len(task["test"])):
        cands = []
        if dsl_out.get(tid):
            cands.append(dsl_out[tid][i]["attempt_1"])
        if tid in ca:
            cands += [ca[tid]["attempts"][i]["attempt_1"], ca[tid]["attempts"][i]["attempt_2"]]
        uniq = []
        for c in cands:
            if c not in uniq:
                uniq.append(c)
        uniq = (uniq + [[[0]], [[0]]])[:2]
        out.append({"attempt_1": uniq[0], "attempt_2": uniq[1]})
    submission[tid] = out
json.dump(submission, open("/kaggle/working/submission.json", "w"))
recs = [json.loads(l) for f in glob.glob("/kaggle/working/ca/*.jsonl") for l in open(f)]
errors = sum("error" in r for r in recs)
fallbacks = sum(r.get("mode") == "eager" and "attempts" in r for r in recs)
steps = sorted(r.get("steps", 0) for r in ca.values())
print(f"wrote {len(submission)} tasks | CompressARC done {len(ca)} | eager fallbacks {fallbacks} | errors {errors} | "
      f"steps min/median/max {steps[0] if steps else 0}/{steps[len(steps)//2] if steps else 0}/{steps[-1] if steps else 0} | "
      f"{(time.time()-T0)/3600:.2f}h", flush=True)
'''


def build(u: str) -> Path:
    d = BUILD / "submission"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "run.py").write_text(RUNNER_V2)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{u}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": False,
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
    elif cmd == "pull":
        out = Path(__file__).resolve().parents[1] / "runs" / "submission_run"
        out.mkdir(parents=True, exist_ok=True)
        print(kg("kernels", "output", f"{u}/{KERNEL_SLUG}", "-p", str(out), "--force", check=False).strip()[-300:])
    elif cmd == "submit":
        version, message = sys.argv[2], sys.argv[3]
        print(kg("competitions", "submit", COMP, "-k", f"{u}/{KERNEL_SLUG}", "-v", version,
                 "-f", "submission.json", "-m", message).strip())
