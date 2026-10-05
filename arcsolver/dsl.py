"""Grid -> grid primitives. Every primitive returns a new array or None if not applicable."""
from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

import numpy as np

Grid = np.ndarray
Prim = Callable[[Grid], Optional[Grid]]


def background(g: Grid) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(counts)])


def bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0:
        return None
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def objects(g: Grid, diag: bool = True, by_color: bool = True) -> list[np.ndarray]:
    """Connected components of non-background cells, as boolean masks."""
    bg = background(g)
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    out = []
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag:
        nbrs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] == bg:
                continue
            mask = np.zeros_like(g, dtype=bool)
            stack = [(r, c)]
            seen[r, c] = True
            col = g[r, c]
            while stack:
                y, x = stack.pop()
                mask[y, x] = True
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] != bg:
                        if by_color and g[ny, nx] != col:
                            continue
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append(mask)
    return out


def crop_mask(g: Grid, mask: np.ndarray) -> Optional[Grid]:
    b = bbox(mask)
    if b is None:
        return None
    r0, r1, c0, c1 = b
    return g[r0:r1, c0:c1].copy()


# ---------- geometric ----------

def rot90(g): return np.rot90(g, -1).copy()
def rot180(g): return np.rot90(g, 2).copy()
def rot270(g): return np.rot90(g, 1).copy()
def flip_h(g): return g[:, ::-1].copy()
def flip_v(g): return g[::-1, :].copy()
def transpose(g): return g.T.copy()
def anti_transpose(g): return np.rot90(g, 2).T.copy()


# ---------- cropping / selection ----------

def crop_nonbg(g):
    bg = background(g)
    return crop_mask(g, g != bg)


def _obj_select(g, key, diag=True, by_color=True):
    objs = objects(g, diag=diag, by_color=by_color)
    if len(objs) < 2:
        return None
    return crop_mask(g, key(objs))


def largest_obj(g): return _obj_select(g, lambda o: max(o, key=lambda m: m.sum()))
def smallest_obj(g): return _obj_select(g, lambda o: min(o, key=lambda m: m.sum()))
def largest_multi(g): return _obj_select(g, lambda o: max(o, key=lambda m: m.sum()), by_color=False)
def smallest_multi(g): return _obj_select(g, lambda o: min(o, key=lambda m: m.sum()), by_color=False)


def unique_shape_obj(g):
    """Object whose (cropped) pattern appears exactly once."""
    objs = objects(g, by_color=False)
    if len(objs) < 3:
        return None
    crops = [crop_mask(g, m) for m in objs]
    keys = [c.tobytes() + bytes(c.shape) for c in crops]
    cnt = Counter(keys)
    singles = [c for c, k in zip(crops, keys) if cnt[k] == 1]
    return singles[0] if len(singles) == 1 else None


def most_colors_obj(g):
    objs = objects(g, by_color=False)
    if len(objs) < 2:
        return None
    best = max(objs, key=lambda m: len(np.unique(g[m])))
    return crop_mask(g, best)


def top_half(g): return g[: g.shape[0] // 2].copy() if g.shape[0] >= 2 else None
def bottom_half(g): return g[(g.shape[0] + 1) // 2:].copy() if g.shape[0] >= 2 else None
def left_half(g): return g[:, : g.shape[1] // 2].copy() if g.shape[1] >= 2 else None
def right_half(g): return g[:, (g.shape[1] + 1) // 2:].copy() if g.shape[1] >= 2 else None


# ---------- scaling / tiling ----------

def _upscale(k):
    def f(g):
        if g.shape[0] * k > 30 or g.shape[1] * k > 30:
            return None
        return np.kron(g, np.ones((k, k), dtype=g.dtype))
    f.__name__ = f"upscale{k}"
    return f


def _downscale(k):
    def f(g):
        h, w = g.shape
        if h % k or w % k:
            return None
        small = g[::k, ::k]
        if not np.array_equal(np.kron(small, np.ones((k, k), dtype=g.dtype)), g):
            return None
        return small.copy()
    f.__name__ = f"downscale{k}"
    return f


def _tile(ry, rx):
    def f(g):
        if g.shape[0] * ry > 30 or g.shape[1] * rx > 30:
            return None
        return np.tile(g, (ry, rx))
    f.__name__ = f"tile{ry}x{rx}"
    return f


def mirror_tile_2x2(g):
    if g.shape[0] * 2 > 30 or g.shape[1] * 2 > 30:
        return None
    top = np.hstack([g, g[:, ::-1]])
    return np.vstack([top, top[::-1]])


def mirror_h_concat(g):
    return np.hstack([g, g[:, ::-1]]) if g.shape[1] * 2 <= 30 else None


def mirror_v_concat(g):
    return np.vstack([g, g[::-1]]) if g.shape[0] * 2 <= 30 else None


def self_kron(g):
    """Fractal: place g wherever g is non-background."""
    h, w = g.shape
    if h * h > 30 or w * w > 30:
        return None
    bg = background(g)
    out = np.full((h * h, w * w), bg, dtype=g.dtype)
    for r in range(h):
        for c in range(w):
            if g[r, c] != bg:
                out[r * h:(r + 1) * h, c * w:(c + 1) * w] = g
    return out


# ---------- physics / filling ----------

def _gravity(direction):
    def f(g):
        bg = background(g)
        x = np.rot90(g, direction).copy()
        out = np.full_like(x, bg)
        for c in range(x.shape[1]):
            col = x[:, c]
            vals = col[col != bg]
            if len(vals):
                out[-len(vals):, c] = vals
        return np.rot90(out, -direction).copy()
    f.__name__ = f"gravity{direction}"
    return f


def fill_enclosed(g):
    """Fill background regions not connected to the border with the most common non-bg color."""
    bg = background(g)
    h, w = g.shape
    outside = np.zeros_like(g, dtype=bool)
    stack = [(r, c) for r in range(h) for c in (0, w - 1)] + [(r, c) for c in range(w) for r in (0, h - 1)]
    while stack:
        y, x = stack.pop()
        if 0 <= y < h and 0 <= x < w and not outside[y, x] and g[y, x] == bg:
            outside[y, x] = True
            stack += [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
    hole = (g == bg) & ~outside
    if not hole.any():
        return None
    fg = g[g != bg]
    out = g.copy()
    out[hole] = Counter(fg.tolist()).most_common(1)[0][0]
    return out


def symmetrize(g):
    """Complete a partially-occluded symmetric pattern by overlaying its mirror images (bg = holes)."""
    bg = background(g)
    out = g.copy()
    for m in (g[:, ::-1], g[::-1, :], g[::-1, ::-1]):
        hole = out == bg
        out[hole] = m[hole]
    return None if np.array_equal(out, g) else out


def _logic(op, axis):
    """Split grid into two halves (optionally with a separator line) and combine masks."""
    def f(g):
        x = g if axis == 0 else g.T
        h = x.shape[0]
        if h % 2 == 1:
            a, b = x[: h // 2], x[h // 2 + 1:]
        else:
            a, b = x[: h // 2], x[h // 2:]
        if a.shape != b.shape:
            return None
        ma, mb = a != 0, b != 0
        m = {"and": ma & mb, "or": ma | mb, "xor": ma ^ mb, "nor": ~(ma | mb)}[op]
        out = m.astype(g.dtype)  # color fixed later by color-map stage
        return out if axis == 0 else out.T
    f.__name__ = f"logic_{op}_{axis}"
    return f


def remove_noise(g):
    """Drop single-cell objects."""
    bg = background(g)
    out = g.copy()
    for m in objects(g, diag=False, by_color=True):
        if m.sum() == 1:
            out[m] = bg
    return None if np.array_equal(out, g) else out


def outline(g):
    """Keep only object borders."""
    bg = background(g)
    fg = g != bg
    p = np.pad(fg, 1)
    interior = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:] & fg
    if not interior.any():
        return None
    out = g.copy()
    out[interior] = bg
    return out


PRIMITIVES: list[Prim] = [
    rot90, rot180, rot270, flip_h, flip_v, transpose, anti_transpose,
    crop_nonbg, largest_obj, smallest_obj, largest_multi, smallest_multi,
    unique_shape_obj, most_colors_obj,
    top_half, bottom_half, left_half, right_half,
    _upscale(2), _upscale(3), _downscale(2), _downscale(3),
    _tile(1, 2), _tile(2, 1), _tile(2, 2), _tile(3, 3),
    mirror_tile_2x2, mirror_h_concat, mirror_v_concat, self_kron,
    _gravity(0), _gravity(1), _gravity(2), _gravity(3),
    fill_enclosed, symmetrize, remove_noise, outline,
    *[_logic(op, ax) for op in ("and", "or", "xor", "nor") for ax in (0, 1)],
]
