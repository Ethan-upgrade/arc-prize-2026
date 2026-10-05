"""Run the hint A/B experiment on Kaggle CPU kernels.

    python scripts/kaggle_ab.py push     # upload code+data as a private dataset, start the kernels
    python scripts/kaggle_ab.py status   # kernel states
    python scripts/kaggle_ab.py pull     # download results into runs/ab_kaggle/

Needs a Kaggle API token in ~/.kaggle/. Nothing here prints the key.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "kaggle"
KAGGLE = str(ROOT / ".venv" / "Scripts" / "kaggle")
N_KERNELS, PROCS_PER_KERNEL, STEPS = 2, 4, 500
DATASET_SLUG = "arc2026-hint-ab-code"

RUNNER = '''
import glob, os, shutil, subprocess, sys
# Dataset mount paths have changed across Kaggle versions; find our package wherever it is.
SRC = os.path.dirname(sorted(glob.glob("/kaggle/input/**/arcsolver/ab.py", recursive=True))[0])
SRC = os.path.dirname(SRC)
WORK = "/kaggle/working/src"
shutil.copytree(SRC, WORK, dirs_exist_ok=True)
os.makedirs("/kaggle/working/ab", exist_ok=True)
procs = []
for j in range({ppk}):
    shard = {kernel} * {ppk} + j
    out = f"/kaggle/working/ab/shard{{shard}}"
    procs.append(subprocess.Popen(
        [sys.executable, "-u", "-m", "arcsolver.ab", "--steps", "{steps}",
         "--shard", f"{{shard}}/{total}", "--out", out + ".jsonl"],
        cwd=WORK, env={{**os.environ, "DEV": "cpu", "THREADS": "1"}},
        stdout=open(out + ".log", "w"), stderr=subprocess.STDOUT))
codes = [p.wait() for p in procs]
print("exit codes", codes)
for f in sorted(os.listdir("/kaggle/working/ab")):
    if f.endswith(".log"):
        print("==", f); print(open("/kaggle/working/ab/" + f).read()[-2000:])
shutil.rmtree(WORK)  # keep only results in the output
'''


def user() -> str:
    r = subprocess.run([KAGGLE, "config", "view"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.strip().startswith("- username:"):
            name = line.split(":", 1)[1].strip()
            if name and name != "None":
                return name
    sys.exit("Kaggle CLI is not authenticated: put a token in ~/.kaggle/access_token")


def kg(*args: str, check: bool = True) -> str:
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"kaggle {' '.join(args[:2])} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def build_dataset(u: str) -> Path:
    d = BUILD / "dataset"
    shutil.rmtree(d, ignore_errors=True)
    (d / "arcsolver").mkdir(parents=True)
    for f in (ROOT / "arcsolver").glob("*.py"):
        shutil.copy(f, d / "arcsolver" / f.name)
    ca_src, ca_dst = ROOT / "third_party" / "CompressARC", d / "third_party" / "CompressARC"
    (ca_dst / "dataset").mkdir(parents=True)
    for f in ca_src.glob("*.py"):
        if f.name != "bench_one.py":
            shutil.copy(f, ca_dst / f.name)
    shutil.copy(ca_src / "LICENSE", ca_dst / "LICENSE")
    for f in ca_src.glob("dataset/arc-agi_agi2eval_*.json"):
        shutil.copy(f, ca_dst / "dataset" / f.name)
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": "ARC 2026 hint AB code", "id": f"{u}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}]}))
    return d


def build_kernel(u: str, k: int) -> Path:
    d = BUILD / f"kernel{k}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "run.py").write_text(RUNNER.format(ppk=PROCS_PER_KERNEL, kernel=k,
                                            steps=STEPS, total=N_KERNELS * PROCS_PER_KERNEL))
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{u}/arc2026-hint-ab-{k}", "title": f"arc2026-hint-ab-{k}",
        "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [f"{u}/{DATASET_SLUG}"], "competition_sources": [],
        "kernel_sources": []}))
    return d


def push():
    u = user()
    ds = build_dataset(u)
    exists = "403" not in (s := kg("datasets", "status", f"{u}/{DATASET_SLUG}", check=False)) \
        and "404" not in s and "error" not in s.lower()
    if exists:
        kg("datasets", "version", "-p", str(ds), "-m", time.strftime("%Y-%m-%d %H:%M"), "-r", "zip")
    else:
        kg("datasets", "create", "-p", str(ds), "-r", "zip")
    print("dataset uploaded; waiting for it to be ready ...")
    for _ in range(60):
        if "ready" in kg("datasets", "status", f"{u}/{DATASET_SLUG}", check=False).lower():
            break
        time.sleep(10)
    for k in range(N_KERNELS):
        print(kg("kernels", "push", "-p", str(build_kernel(u, k))).strip())


def status():
    u = user()
    for k in range(N_KERNELS):
        print(kg("kernels", "status", f"{u}/arc2026-hint-ab-{k}", check=False).strip())


def pull():
    u = user()
    out = ROOT / "runs" / "ab_kaggle"
    out.mkdir(parents=True, exist_ok=True)
    for k in range(N_KERNELS):
        print(kg("kernels", "output", f"{u}/arc2026-hint-ab-{k}", "-p", str(out), check=False).strip()[-300:])


if __name__ == "__main__":
    {"push": push, "status": status, "pull": pull}[sys.argv[1]]()
