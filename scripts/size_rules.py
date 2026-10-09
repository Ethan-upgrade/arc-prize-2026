"""Can a cheap symbolic rule predict the test output SIZE where CompressARC gets it wrong?

    python scripts/size_rules.py

A rule maps a task's input grid to an output (h, w). It is accepted only if it is exact on every
training pair; then we check it on the test outputs. No GPU, no model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arcsolver.dsl import background, bbox, objects  # noqa: E402

DATA = ROOT / "third_party" / "CompressARC" / "dataset"


def _objs(g, by_color):
    return objects(g, diag=True, by_color=by_color)


def _box(mask):
    b = bbox(mask)
    return (b[1] - b[0], b[3] - b[2]) if b else None


def rules():
    """name -> fn(input grid) -> (h, w) or None."""
    R = {}
    R["same"] = lambda g: g.shape
    R["transpose"] = lambda g: g.shape[::-1]
    for k in (2, 3):
        R[f"x{k}"] = lambda g, k=k: (g.shape[0] * k, g.shape[1] * k)
        R[f"div{k}"] = lambda g, k=k: (g.shape[0] // k, g.shape[1] // k) if g.shape[0] % k == 0 and g.shape[1] % k == 0 else None
    R["half_h"] = lambda g: (g.shape[0] // 2, g.shape[1])
    R["half_w"] = lambda g: (g.shape[0], g.shape[1] // 2)
    R["nonbg_bbox"] = lambda g: _box(g != background(g))
    for by_color in (True, False):
        tag = "mono" if by_color else "multi"
        R[f"largest_{tag}"] = lambda g, b=by_color: _box(max(_objs(g, b), key=lambda m: m.sum())) if _objs(g, b) else None
        R[f"smallest_{tag}"] = lambda g, b=by_color: _box(min(_objs(g, b), key=lambda m: m.sum())) if _objs(g, b) else None
        R[f"n_objs_{tag}_x1"] = lambda g, b=by_color: (len(_objs(g, b)), 1)
        R[f"1x_n_objs_{tag}"] = lambda g, b=by_color: (1, len(_objs(g, b)))
    R["n_colors_sq"] = lambda g: (len(np.unique(g)) - 1,) * 2
    R["rarest_color_bbox"] = lambda g: _box(g == min(np.unique(g), key=lambda c: (g == c).sum()))

    def separators(g):
        """Grid split by full-length lines of one colour: size of one cell."""
        h, w = g.shape
        rows = [r for r in range(h) if len(np.unique(g[r])) == 1 and g[r, 0] != background(g)]
        cols = [c for c in range(w) if len(np.unique(g[:, c])) == 1 and g[0, c] != background(g)]
        if not rows and not cols:
            return None
        def seg(lines, n):
            cuts = [-1] + lines + [n]
            sizes = [b - a - 1 for a, b in zip(cuts, cuts[1:]) if b - a - 1 > 0]
            return sizes[0] if sizes and len(set(sizes)) == 1 else None
        sh, sw = seg(rows, h), seg(cols, w)
        return (sh, sw) if sh and sw else None
    R["separator_cell"] = separators
    return R


def fit(task):
    """Rules exact on all train pairs, plus a constant-size rule."""
    ins = [np.array(p["input"]) for p in task["train"]]
    outs = [np.array(p["output"]) for p in task["train"]]
    ok = []
    for name, fn in rules().items():
        try:
            if all(fn(i) == o.shape for i, o in zip(ins, outs)):
                ok.append((name, fn))
        except Exception:
            pass
    if len({o.shape for o in outs}) == 1:
        const = outs[0].shape
        ok.append(("constant", lambda g, c=const: c))
    return ok


def main():
    ch = json.load(open(DATA / "arc-agi_evaluation_challenges.json"))
    sol = json.load(open(DATA / "arc-agi_evaluation_solutions.json"))
    fa = json.load(open(ROOT / "runs" / "failure_analysis.json"))
    ws = {(r["task"], r["i"]) for r in fa if r["cat"] == "wrong_shape"}

    def evaluate(keys):
        covered = right = ambiguous = 0
        for t, i in keys:
            ok = fit(ch[t])
            if not ok:
                continue
            covered += 1
            g = np.array(ch[t]["test"][i]["input"])
            preds = set()
            for _, fn in ok:
                try:
                    p = fn(g)
                    if p:
                        preds.add(tuple(p))
                except Exception:
                    pass
            truth = np.array(sol[t][i]).shape
            if len(preds) > 1:
                ambiguous += 1
            right += truth in preds and len(preds) <= 2  # usable as one of two attempts
        return covered, right, ambiguous

    print("On CompressARC's wrong_shape outputs:")
    c, r, a = evaluate(sorted(ws))
    print(f"  {len(ws)} outputs | a rule fits train: {c} | truth among <=2 predicted sizes: {r} | ambiguous: {a}")

    all_keys = [(t, i) for t in ch for i in range(len(sol[t]))]
    non_same = [(t, i) for t, i in all_keys
                if not all(np.array(p["input"]).shape == np.array(p["output"]).shape for p in ch[t]["train"])]
    c, r, a = evaluate(non_same)
    print(f"\nAll ARC-AGI-1 eval outputs whose task changes size: {len(non_same)}")
    print(f"  a rule fits train: {c} | truth among <=2 predicted sizes: {r} ({100 * r / len(non_same):.0f}%) | ambiguous: {a}")


if __name__ == "__main__":
    main()
