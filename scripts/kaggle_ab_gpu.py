"""Hint A/B on ARC-AGI-1 evaluation, on one Kaggle GPU with several processes sharing it.

    python scripts/kaggle_ab_gpu.py push     # (re)upload code+data, start the GPU kernel
    python scripts/kaggle_ab_gpu.py status
    python scripts/kaggle_ab_gpu.py pull     # -> runs/ab_gpu/

CompressARC is launch-bound (GPU <3% memory, mostly idle), so several processes on the same
GPU raise throughput roughly linearly until the CPU runs out of cores.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_ab import BUILD, DATASET_SLUG, ROOT, build_dataset, kg, user  # noqa: E402

# Round 2 (10-06): fresh tasks 40-79 to test "identity hints hurt" without reusing the tasks it came from.
KERNEL_SLUG = "arc2026-hint-ab-gpu"
TASK_SLICE, ARMS, STEPS, PROCS, N_KERNELS = (40, 80), ["base", "base_s1", "hint_noid"], 300, 4, 2
# Results are flushed per (task, arm), so whatever finishes before Kaggle's limit is kept.
TIME_BUDGET_HOURS = 11

RUNNER = r'''
import glob, json, os, subprocess, sys, time
code = os.path.dirname(os.path.dirname(sorted(glob.glob("/kaggle/input/**/arcsolver/ab.py", recursive=True))[0]))
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout, "cpus", os.cpu_count(), flush=True)
tasks = {tasks}
os.makedirs("/kaggle/working/ab", exist_ok=True)
deadline = time.time() + {budget} * 3600
procs = []
for j in range({procs}):
    out = f"/kaggle/working/ab/proc{{j}}"
    procs.append(subprocess.Popen(
        [sys.executable, "-u", "-m", "arcsolver.ab", "--split", "evaluation", "--steps", "{steps}",
         "--arms", *{arms}, "--tasks", *tasks[j::{procs}], "--out", out + ".jsonl"],
        cwd=code, env={{**os.environ, "DEV": "cuda", "THREADS": "1"}},
        stdout=open(out + ".log", "w"), stderr=subprocess.STDOUT))
while any(p.poll() is None for p in procs) and time.time() < deadline:
    time.sleep(30)
for p in procs:
    if p.poll() is None:
        p.terminate()
print("exit codes", [p.wait() for p in procs], flush=True)
for f in sorted(glob.glob("/kaggle/working/ab/*.log")):
    print("==", f); print(open(f).read()[-1500:])
'''


def slug(k: int) -> str:
    return f"{KERNEL_SLUG}-r2-{k}"


def build_kernel(u: str, tasks: list[str], k: int) -> Path:
    d = BUILD / f"ab_gpu_kernel{k}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "run.py").write_text(RUNNER.format(tasks=tasks, budget=TIME_BUDGET_HOURS, procs=PROCS, steps=STEPS, arms=ARMS))
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{u}/{slug(k)}", "title": slug(k), "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": False,
        "dataset_sources": [f"{u}/{DATASET_SLUG}"], "competition_sources": [],
        "kernel_sources": []}))
    return d


def selected_tasks() -> list[str]:
    sel = json.loads((ROOT / "runs" / "hint_tasks_agi1.json").read_text())
    return [r["task"] for r in sel["tasks"][TASK_SLICE[0]:TASK_SLICE[1]]]


if __name__ == "__main__":
    u = user()
    cmd = sys.argv[1]
    if cmd == "push":
        kg("datasets", "version", "-p", str(build_dataset(u)), "-m", time.strftime("%Y-%m-%d %H:%M"), "-r", "zip")
        for _ in range(60):
            if "ready" in kg("datasets", "status", f"{u}/{DATASET_SLUG}", check=False).lower():
                break
            time.sleep(10)
        tasks = selected_tasks()
        for k in range(N_KERNELS):
            print(kg("kernels", "push", "-p", str(build_kernel(u, tasks[k::N_KERNELS], k))).strip())
    elif cmd == "status":
        for k in range(N_KERNELS):
            print(kg("kernels", "status", f"{u}/{slug(k)}", check=False).strip())
    elif cmd == "pull":
        for k in range(N_KERNELS):
            out = ROOT / "runs" / "ab_gpu_r2" / f"k{k}"
            out.mkdir(parents=True, exist_ok=True)
            print(kg("kernels", "output", f"{u}/{slug(k)}", "-p", str(out), "--force", check=False).strip()[-300:])
