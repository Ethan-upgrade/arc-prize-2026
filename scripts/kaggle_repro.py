"""Reproduce CompressARC (unmodified) on Kaggle GPU before building on it.

    python scripts/kaggle_repro.py push      # upload code+ARC-AGI-1 eval data, start a GPU kernel
    python scripts/kaggle_repro.py status
    python scripts/kaggle_repro.py pull      # results -> runs/repro/

Target tasks are ARC-AGI-1 evaluation tasks that the author's own published run
(results_for_the_blog_post/predictions_evaluation.npz) solved within 200 steps at pass@2.
If we can't solve those, the problem is our setup, not the method.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_ab import BUILD, ROOT, kg, user  # noqa: E402

DATASET_SLUG = "arc2026-repro-code"
KERNEL_SLUG = "arc2026-compressarc-repro"
TASKS = "4cd1b7b2 bbb1b8b6 fc754716 8597cfd7 a406ac07 c7d4e6ad 332efdb3 1a2e2828".split()
STEPS = 400

RUNNER = r'''
import glob, json, os, subprocess, sys, time
src = os.path.dirname(sorted(glob.glob("/kaggle/input/**/arc_compressor.py", recursive=True))[0])
os.chdir(src)
sys.path.insert(0, src)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout, flush=True)
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import arc_compressor, preprocessing, solution_selection, train
assert torch.cuda.is_available()

problems = json.load(open("dataset/arc-agi_evaluation_challenges.json"))
solutions = json.load(open("dataset/arc-agi_evaluation_solutions.json"))
out = open("/kaggle/working/repro.jsonl", "w")
for name in {tasks}:
    torch.manual_seed(0)
    task = preprocessing.Task(name, problems[name], solutions[name])
    model = arc_compressor.ARCCompressor(task)
    opt = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
    log = solution_selection.Logger(task)
    curve, t0 = {{}}, time.time()
    for step in range({steps}):
        train.take_step(task, model, opt, step, log)
        if (step + 1) % 50 == 0:
            a = [log.solution_most_frequent, log.solution_second_most_frequent]
            truth = tuple(tuple(map(tuple, g)) for g in solutions[name])
            curve[step + 1] = int(any(x == truth for x in a))
    torch.cuda.synchronize()
    sec = time.time() - t0
    rec = {{"task": name, "curve": curve, "seconds": round(sec, 1),
           "sec_per_step": round(sec / {steps}, 3),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}}
    out.write(json.dumps(rec) + "\n"); out.flush()
    print(rec, flush=True)
    torch.cuda.reset_peak_memory_stats()
'''


def build_dataset(u: str) -> Path:
    d = BUILD / "repro_dataset"
    shutil.rmtree(d, ignore_errors=True)
    ca_src, ca_dst = ROOT / "third_party" / "CompressARC", d / "CompressARC"
    (ca_dst / "dataset").mkdir(parents=True)
    for f in ca_src.glob("*.py"):
        if f.name != "bench_one.py":
            shutil.copy(f, ca_dst / f.name)
    shutil.copy(ca_src / "LICENSE", ca_dst / "LICENSE")
    for f in ("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"):
        shutil.copy(ca_src / "dataset" / f, ca_dst / "dataset" / f)
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": "ARC 2026 CompressARC repro", "id": f"{u}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}]}))
    return d


def build_kernel(u: str) -> Path:
    d = BUILD / "repro_kernel"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "run.py").write_text(RUNNER.format(tasks=TASKS, steps=STEPS))
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{u}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": False,
        "dataset_sources": [f"{u}/{DATASET_SLUG}"], "competition_sources": [],
        "kernel_sources": []}))
    return d


if __name__ == "__main__":
    u = user()
    cmd = sys.argv[1]
    if cmd == "push":
        kg("datasets", "create", "-p", str(build_dataset(u)), "-r", "zip")
        import time
        for _ in range(60):
            if "ready" in kg("datasets", "status", f"{u}/{DATASET_SLUG}", check=False).lower():
                break
            time.sleep(10)
        print(kg("kernels", "push", "-p", str(build_kernel(u))).strip())
    elif cmd == "status":
        print(kg("kernels", "status", f"{u}/{KERNEL_SLUG}", check=False).strip())
    elif cmd == "pull":
        out = ROOT / "runs" / "repro"
        out.mkdir(parents=True, exist_ok=True)
        print(kg("kernels", "output", f"{u}/{KERNEL_SLUG}", "-p", str(out), "--force", check=False).strip()[-300:])
