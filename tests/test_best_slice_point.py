import sys, random, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
torch.set_default_device("cpu")
from arcsolver.hinted import COMPRESSARC
import solution_selection
orig = solution_selection.Logger._best_slice_point
import arcsolver.fast as fast
new = fast._fast_best_slice_point
class T: pass
random.seed(0); torch.manual_seed(0)
bad = 0; N = 4000
for k in range(N):
    n = random.randint(1, 30)
    t = T(); t.in_out_same_size = t.all_out_same_size = (k % 3 == 0)
    o = type("L", (), {})(); o.task = t
    kind = k % 4
    if kind == 0: mask = torch.randn(n) * 50
    elif kind == 1: mask = torch.round(torch.randn(n)) * 1000          # many exact ties
    elif kind == 2: mask = torch.full((n,), -1000.0); mask[:random.randint(0, n)] = 1000  # mask-like
    else: mask = torch.zeros(n)                                           # all ties
    L = random.randint(1, n)
    a = orig(o, mask, L); b = new(o, mask, L)
    if tuple(a) != tuple(b):
        bad += 1
        if bad <= 5: print("mismatch", kind, n, L, a, b)
print(f"{N} random masks: mismatches {bad}")
