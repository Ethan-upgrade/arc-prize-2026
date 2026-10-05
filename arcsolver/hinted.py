"""CompressARC with symbolic hint channels.

Each hint is the output of a (possibly imperfect) DSL program found by `partial_search`.
It enters the decoder as an additive logit bias `gate_k * onehot(hint_k)` on the output grids,
with one learnable scalar gate per hint, initialised at 0. In MDL terms, the per-task model now
only has to encode the residual between the hint and the target, and a useless hint costs
(almost) nothing because its gate stays near zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

COMPRESSARC = Path(__file__).resolve().parents[1] / "third_party" / "CompressARC"
sys.path.insert(0, str(COMPRESSARC))

import arc_compressor  # noqa: E402

from .search import Hint  # noqa: E402


def hint_tensor(task, hint: Hint) -> torch.Tensor | None:
    """[example, n_colors, n_x, n_y] bias for the non-black logit channels, or None if unusable.

    Adding +1 to colour c's logit, or -1 to every non-black logit when the hint is black, is
    softmax-equivalent to a +1 on the hinted colour of the full (black-included) distribution.
    """
    grids = list(hint.train_outputs) + list(hint.test_outputs)
    if len(grids) != task.n_examples:
        return None
    bias = np.zeros((task.n_examples, task.n_colors, task.n_x, task.n_y), dtype=np.float32)
    for e, g in enumerate(grids):
        h, w = g.shape
        if h > task.n_x or w > task.n_y:
            return None
        for color in np.unique(g).tolist():
            if color not in task.colors:
                return None  # hint invents a colour the task never uses
            idx = task.colors.index(color)
            m = g == color
            if idx == 0:
                bias[e, :, :h, :w][:, m] -= 1.0
            else:
                bias[e, idx - 1, :h, :w][m] += 1.0
    return torch.from_numpy(bias).to(torch.get_default_device())


class HintedCompressor(arc_compressor.ARCCompressor):
    def __init__(self, task, hints: list[Hint]):
        super().__init__(task)
        self.hint_biases = [b for b in (hint_tensor(task, h) for h in hints) if b is not None]
        self.gates = [torch.zeros((), requires_grad=True) for _ in self.hint_biases]
        self.weights_list = self.weights_list + self.gates

    def forward(self):
        output, x_mask, y_mask, KL_amounts, KL_names = super().forward()
        if self.hint_biases:
            out_logits = output[..., 1]
            for gate, bias in zip(self.gates, self.hint_biases):
                out_logits = out_logits + gate * bias
            output = torch.stack([output[..., 0], out_logits], dim=-1)
        return output, x_mask, y_mask, KL_amounts, KL_names
