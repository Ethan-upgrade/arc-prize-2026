"""Check the prefix-sum window scorer against train.mask_select_logprobs, elementwise.

The training graph uses arcsolver.fast._fast_mask_select_logprobs instead of the original
three-reductions-per-offset version. The loss is only equivalent if both return the same
partition and the same per-offset scores, so compare them on random mask-like vectors,
including the tie-heavy and constant cases that make exactness awkward.

    python tests/test_fast_logprobs.py
"""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_default_device("cpu")

from arcsolver.hinted import COMPRESSARC  # noqa: F401,E402  (puts CompressARC on sys.path)
import train  # noqa: E402
original_mask_select_logprobs = train.mask_select_logprobs

import arcsolver.fast as fast  # noqa: E402


def main():
    random.seed(0)
    torch.manual_seed(0)
    bad = 0
    worst = 0.0
    N = 2000
    for k in range(N):
        n = random.randint(1, 30)
        kind = k % 4
        if kind == 0:
            mask = torch.randn(n) * 5
        elif kind == 1:
            mask = torch.round(torch.randn(n)) * 1000
        elif kind == 2:
            mask = torch.full((n,), -1000.0)
            mask[: random.randint(0, n)] = 1000
        else:
            mask = torch.zeros(n)
        mask = mask.to(torch.float32)
        L = random.randint(1, n)

        part_ref, lp_ref = original_mask_select_logprobs(mask, L)
        part_new, lp_new = fast._fast_mask_select_logprobs(mask, L)

        d_score = float((lp_ref - lp_new).abs().max())
        d_part = float((part_ref - part_new).abs())
        worst = max(worst, d_score, d_part)
        # The prefix formula changes float32 summation order relative to three separate sums.
        # Independent-reference runs observed a maximum absolute difference of 1.53e-5.
        if lp_ref.shape != lp_new.shape or d_score > 2e-5 or d_part > 2e-5:
            bad += 1
            if bad <= 5:
                print(f"mismatch kind={kind} n={n} L={L} d_score={d_score:.3e} d_part={d_part:.3e}")

    print(f"{N} random masks: mismatches {bad}, worst abs diff {worst:.2e}")
    assert bad == 0, f"{bad}/{N} fixed-length comparisons exceeded tolerance"

    # The uncertain-size path uses a second helper that must match the same contract:
    # return (log_partition, logprobs), where the partition covers EVERY (length, offset) window.
    bad2 = 0
    worst2 = 0.0
    for k in range(N):
        n = random.randint(1, 30)
        mask = torch.rand(n, dtype=torch.float32) * 10
        L = random.randint(1, n)

        part_ref, lp_ref = original_mask_select_logprobs(mask, L)
        all_parts = [original_mask_select_logprobs(mask, m)[0] for m in range(1, n + 1)]
        all_ref = torch.logsumexp(torch.stack(all_parts), dim=0)

        part_new, lp_new = fast._fast_window_scores_and_partition(mask, L)

        d_lp = float((lp_ref - lp_new).abs().max())
        d_ap = float((all_ref - part_new).abs())
        worst2 = max(worst2, d_lp, d_ap)
        # Both fixed scores and the all-size partition use reordered float32 reductions.
        # The independent reference reaches 4.58e-5 absolute difference for masks scaled to 10.
        if d_lp > 1e-4 or d_ap > 1e-4:
            bad2 += 1
            if bad2 <= 5:
                print(f"partition mismatch n={n} L={L} d_lp={d_lp:.3e} d_partition={d_ap:.3e}")

    print(f"{N} random masks (all-size partition): mismatches {bad2}, worst abs diff {worst2:.2e}")
    assert bad2 == 0, f"{bad2}/{N} all-size partition comparisons exceeded tolerance"


if __name__ == "__main__":
    main()
