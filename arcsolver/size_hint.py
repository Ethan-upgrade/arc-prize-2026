"""Symbolic output-size hints for CompressARC.

CompressARC already decodes with a fixed crop length when the output size is known (same-size or
constant-size tasks). For the remaining tasks it searches over every length, and that is where it
most often gets the size wrong. `predict_sizes` fits simple size rules on the training pairs; when
they agree on one size for a test input, `SizeHintLogger` decodes with that length instead of
searching. Training is untouched: only how an answer is read out of the model changes.
"""
from __future__ import annotations

import numpy as np

from .dsl import background, bbox, objects
from .hinted import COMPRESSARC  # noqa: F401  (puts CompressARC on sys.path)
import solution_selection  # noqa: E402


def _box(mask):
    b = bbox(mask)
    return (int(b[1] - b[0]), int(b[3] - b[2])) if b else None


def _objs(g, by_color):
    return objects(g, diag=True, by_color=by_color)


def _separator_cell(g):
    """Grid split by full-length single-colour lines: the size of one cell, if all cells match."""
    h, w = g.shape
    bg = background(g)
    rows = [r for r in range(h) if len(np.unique(g[r])) == 1 and g[r, 0] != bg]
    cols = [c for c in range(w) if len(np.unique(g[:, c])) == 1 and g[0, c] != bg]
    if not rows and not cols:
        return None

    def seg(lines, n):
        cuts = [-1] + lines + [n]
        sizes = [b - a - 1 for a, b in zip(cuts, cuts[1:]) if b - a - 1 > 0]
        return sizes[0] if sizes and len(set(sizes)) == 1 else None

    sh, sw = seg(rows, h), seg(cols, w)
    return (sh, sw) if sh and sw else None


def size_rules():
    """name -> fn(input grid) -> (h, w) or None. Deliberately small and cheap."""
    R = {
        "transpose": lambda g: g.shape[::-1],
        "half_h": lambda g: (g.shape[0] // 2, g.shape[1]),
        "half_w": lambda g: (g.shape[0], g.shape[1] // 2),
        "nonbg_bbox": lambda g: _box(g != background(g)),
        "n_colors_sq": lambda g: (len(np.unique(g)) - 1,) * 2,
        "rarest_color_bbox": lambda g: _box(g == min(np.unique(g), key=lambda c: (g == c).sum())),
        "separator_cell": _separator_cell,
    }
    for k in (2, 3):
        R[f"x{k}"] = lambda g, k=k: (g.shape[0] * k, g.shape[1] * k)
        R[f"div{k}"] = lambda g, k=k: ((g.shape[0] // k, g.shape[1] // k)
                                       if g.shape[0] % k == 0 and g.shape[1] % k == 0 else None)
    for by_color in (True, False):
        tag = "mono" if by_color else "multi"
        R[f"largest_{tag}"] = lambda g, b=by_color: (_box(max(_objs(g, b), key=lambda m: m.sum()))
                                                     if _objs(g, b) else None)
        R[f"smallest_{tag}"] = lambda g, b=by_color: (_box(min(_objs(g, b), key=lambda m: m.sum()))
                                                      if _objs(g, b) else None)
        R[f"n_objs_{tag}_x1"] = lambda g, b=by_color: (len(_objs(g, b)), 1)
        R[f"1x_n_objs_{tag}"] = lambda g, b=by_color: (1, len(_objs(g, b)))
    return R


def predict_sizes(problem: dict) -> dict[int, tuple[int, int]]:
    """{test index: (h, w)} where every rule that is exact on all train pairs agrees on one size.

    Returns {} for tasks CompressARC already treats as fixed-size, since there is nothing to add.
    """
    ins = [np.array(p["input"]) for p in problem["train"]]
    outs = [np.array(p["output"]) for p in problem["train"]]
    if all(i.shape == o.shape for i, o in zip(ins, outs)) or len({o.shape for o in outs}) == 1:
        return {}
    fitted = []
    for fn in size_rules().values():
        try:
            if all(fn(i) == o.shape for i, o in zip(ins, outs)):
                fitted.append(fn)
        except Exception:
            pass
    hints = {}
    for t, ex in enumerate(problem["test"]):
        g = np.array(ex["input"])
        preds = set()
        for fn in fitted:
            try:
                p = fn(g)
            except Exception:
                p = None
            if p and p[0] > 0 and p[1] > 0:
                preds.add(tuple(int(v) for v in p))
        if len(preds) == 1:
            hints[t] = preds.pop()
    return hints


class SizeHintLogger(solution_selection.Logger):
    """Logger that decodes hinted test outputs with a fixed crop length."""

    def __init__(self, task, size_hints: dict[int, tuple[int, int]]):
        super().__init__(task)
        # Drop hints the model's canvas cannot hold.
        self.size_hints = {i: s for i, s in size_hints.items() if s[0] <= task.n_x and s[1] <= task.n_y}

    def _best_slice_point(self, mask, length):
        if length is not None and not (self.task.in_out_same_size or self.task.all_out_same_size):
            logprobs = [-mask[:o].sum() + mask[o:o + length].sum() - mask[o + length:].sum()
                        for o in range(mask.shape[0] - length + 1)]
            start = int(np.argmax([float(lp) for lp in logprobs]))
            return start, start + length
        return super()._best_slice_point(mask, length)

    def _postprocess_solution(self, prediction, x_mask, y_mask):
        if not self.size_hints:
            return super()._postprocess_solution(prediction, x_mask, y_mask)
        import torch
        colors = torch.argmax(prediction, dim=1)
        uncertainties = torch.logsumexp(prediction, dim=1) - torch.amax(prediction, dim=1)
        slices, unc = [], []
        for e in range(self.task.n_test):
            hx = hy = None
            if self.task.in_out_same_size or self.task.all_out_same_size:
                hx, hy = self.task.shapes[self.task.n_train + e][1]
            elif e in self.size_hints:
                hx, hy = self.size_hints[e]
            sol = self.best_crop(colors[e], x_mask[e], hx, y_mask[e], hy)
            u = self.best_crop(uncertainties[e], x_mask[e], hx, y_mask[e], hy)
            slices.append(sol.cpu().numpy().tolist())
            unc.append(float(np.mean(u.cpu().numpy())))
        for ex in slices:
            for row in ex:
                for j, v in enumerate(row):
                    row[j] = self.task.colors[v]
        return tuple(tuple(tuple(r) for r in ex) for ex in slices), float(np.mean(unc))
