"""End-to-end loss equivalence: arcsolver.fast.loss_terms must match train.take_step.

This is the test that was missing when a swapped return value survived the unit tests: the
per-mask helpers matched the original, but the *caller* unpacked them backwards, so variable
output-size tasks got a scalar where a vector was expected. Comparing whole losses on real
tasks catches that, because the original train.take_step is the reference.

    python tests/test_loss_equivalence.py
"""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arcsolver.hinted import COMPRESSARC  # noqa: E402

os.chdir(COMPRESSARC)
torch.set_default_device("cpu")
import arc_compressor  # noqa: E402
import preprocessing  # noqa: E402
import train  # noqa: E402
original_mask_select_logprobs = train.mask_select_logprobs

import arcsolver.fast as fast  # noqa: E402

# Must include tasks with VARYING output sizes: those take the `uncertain` branch, and that is
# where the swapped return value lived. 0b148d64 crashed with IndexError before the fix.
TASKS = ["25d8a9c8", "0b148d64", "00d62c1b", "1e0a9b12"]


def original_loss(task, model, step):
    """Recompute train.take_step's loss without touching the optimizer or logger."""
    logits, x_mask, y_mask, KL_amounts, KL_names = model.forward()
    logits = torch.cat([torch.zeros_like(logits[:, :1, :, :]), logits], dim=1)
    total_KL = 0
    for a in KL_amounts:
        total_KL = total_KL + torch.sum(a)
    rec = 0
    for e in range(task.n_examples):
        for m in range(2):
            if e >= task.n_train and m == 1:
                continue
            uncertain = not (task.in_out_same_size or task.all_out_same_size and m == 1
                             or task.all_in_same_size and m == 0)
            coeff = 0.01 ** max(0, 1 - step / 100) if uncertain else 1
            out = task.shapes[e][m]
            lp_x, x_lp = original_mask_select_logprobs(coeff * x_mask[e, :, m], out[0])
            lp_y, y_lp = original_mask_select_logprobs(coeff * y_mask[e, :, m], out[1])
            if uncertain:
                lp_x = torch.logsumexp(torch.stack([
                    original_mask_select_logprobs(coeff * x_mask[e, :, m], n)[0]
                    for n in range(1, x_mask.shape[1] + 1)]), dim=0)
                lp_y = torch.logsumexp(torch.stack([
                    original_mask_select_logprobs(coeff * y_mask[e, :, m], n)[0]
                    for n in range(1, y_mask.shape[1] + 1)]), dim=0)
            rows = []
            for xo in range(x_lp.shape[0]):
                row = []
                for yo in range(y_lp.shape[0]):
                    lp = x_lp[xo] - lp_x + y_lp[yo] - lp_y
                    crop = logits[e, :, xo:xo + out[0], yo:yo + out[1], m]
                    target = task.problem[e, :out[0], :out[1], m]
                    lp = lp - torch.nn.functional.cross_entropy(crop[None], target[None], reduction="sum")
                    row.append(lp)
                rows.append(torch.stack(row))
            logprobs = torch.stack(rows)
            ca = 0.1 ** max(0, 1 - step / 100) if uncertain else 1
            rec = rec - torch.logsumexp(ca * logprobs, dim=(0, 1)) / ca
    return (total_KL + 10 * rec).item()


def main():
    problems = json.load(open("dataset/arc-agi_training_challenges.json"))
    sols = json.load(open("dataset/arc-agi_training_solutions.json"))
    bad = 0
    seen = set()
    for name in TASKS:
        if name not in problems:
            print(f"{name}: not in dataset, skipped")
            continue
        seen.add(name)
        torch.manual_seed(0)
        task = preprocessing.Task(name, problems[name], sols[name])
        model = arc_compressor.ARCCompressor(task)
        for step in (0, 50, 250):
            cs, ca = fast.coefficients(step)
            c_size = torch.tensor(cs, dtype=torch.float32)
            c_agg = torch.tensor(ca, dtype=torch.float32)
            # forward() samples noise, so both computations must see the same parameters AND
            # the same sampling. Re-seeding right before each call makes the comparison fair.
            torch.manual_seed(1234)
            ref = original_loss(task, model, step)
            torch.manual_seed(1234)
            got = float(fast.loss_terms(task, model, c_size, c_agg)[0])
            rel = abs(ref - got) / max(abs(ref), 1e-9)
            ok = rel < 1e-3
            bad += not ok
            print(f"{name} step {step:3d}: ref {ref:14.4f}  fast {got:14.4f}  rel {rel:.2e}  {'OK' if ok else 'MISMATCH'}")
    missing = set(TASKS) - seen
    assert not missing, f"required fixture tasks missing from dataset: {sorted(missing)}"
    print(f"mismatches: {bad}")
    assert bad == 0, f"{bad} end-to-end loss comparisons exceeded tolerance"


if __name__ == "__main__":
    main()
