"""CUDA-graph training step for CompressARC.

CompressARC is launch-bound: one step issues tens of thousands of tiny kernels while the GPU
sits mostly idle. Capturing forward + backward + Adam into a single CUDA graph and replaying it
removes the per-kernel CPU overhead. The loss is a copy of `train.take_step` with its two
step-dependent warm-up coefficients turned into GPU tensors, so one graph serves every step.
"""
from __future__ import annotations

import inspect

import torch

from .hinted import COMPRESSARC  # noqa: F401  (puts CompressARC on sys.path)
import layers  # noqa: E402
import train  # noqa: E402


def _patch_channel_layer():
    """layers.channel_layer builds two scalar tensors from Python floats on every call, which is
    a host->device copy and is illegal during graph capture. Plain floats broadcast identically."""
    src = inspect.getsource(layers.channel_layer)
    if "torch.tensor(min_capacity)" not in src:
        return
    src = src.replace("    min_capacity = torch.tensor(min_capacity)\n", "")
    src = src.replace("    init_capacity = torch.tensor(init_capacity)\n", "")
    ns = {}
    exec(compile(src, layers.__file__, "exec"), layers.__dict__, ns)
    layers.channel_layer = ns["channel_layer"]


_patch_channel_layer()

_orig_postprocess_mask = layers.postprocess_mask


def _cached_postprocess_mask(task, x_mask, y_mask):
    """The original rebuilds a task-constant modifier in numpy and copies it to the GPU on every
    forward. Compute it once per task (outside capture) and reuse the device tensor."""
    cache = getattr(task, "_mask_modifiers", None)
    if cache is None:
        cache = _orig_postprocess_mask(task, torch.zeros_like(x_mask), torch.zeros_like(y_mask))
        task._mask_modifiers = cache
    return x_mask + cache[0], y_mask + cache[1]


layers.postprocess_mask = _cached_postprocess_mask

mask_select_logprobs = train.mask_select_logprobs


def loss_terms(task, model, c_size, c_agg):
    """Same maths as train.take_step; c_size / c_agg replace the Python-float coefficients."""
    logits, x_mask, y_mask, KL_amounts, KL_names = model.forward()
    logits = torch.cat([torch.zeros_like(logits[:, :1, :, :]), logits], dim=1)
    total_KL = sum(torch.sum(k) for k in KL_amounts)

    reconstruction_error = 0
    for example_num in range(task.n_examples):
        for in_out_mode in range(2):
            if example_num >= task.n_train and in_out_mode == 1:
                continue
            uncertain = not (task.in_out_same_size or task.all_out_same_size and in_out_mode == 1
                             or task.all_in_same_size and in_out_mode == 0)
            cs = c_size if uncertain else 1
            logits_slice = logits[example_num, :, :, :, in_out_mode]
            problem_slice = task.problem[example_num, :, :, in_out_mode]
            output_shape = task.shapes[example_num][in_out_mode]
            x_part, x_lp = mask_select_logprobs(cs * x_mask[example_num, :, in_out_mode], output_shape[0])
            y_part, y_lp = mask_select_logprobs(cs * y_mask[example_num, :, in_out_mode], output_shape[1])
            if uncertain:
                x_part = torch.logsumexp(torch.stack([
                    mask_select_logprobs(cs * x_mask[example_num, :, in_out_mode], n)[0]
                    for n in range(1, x_mask.shape[1] + 1)]), dim=0)
                y_part = torch.logsumexp(torch.stack([
                    mask_select_logprobs(cs * y_mask[example_num, :, in_out_mode], n)[0]
                    for n in range(1, y_mask.shape[1] + 1)]), dim=0)
            rows = []
            for xo in range(x_lp.shape[0]):
                row = []
                for yo in range(y_lp.shape[0]):
                    lp = x_lp[xo] - x_part + y_lp[yo] - y_part
                    crop = logits_slice[:, xo:xo + output_shape[0], yo:yo + output_shape[1]]
                    target = problem_slice[:output_shape[0], :output_shape[1]]
                    lp = lp - torch.nn.functional.cross_entropy(crop[None], target[None], reduction="sum")
                    row.append(lp)
                rows.append(torch.stack(row))
            logprobs = torch.stack(rows)
            ca = c_agg if uncertain else 1
            reconstruction_error = reconstruction_error - torch.logsumexp(ca * logprobs, dim=(0, 1)) / ca

    loss = total_KL + 10 * reconstruction_error
    return loss, logits, x_mask, y_mask, KL_amounts, KL_names, total_KL, reconstruction_error


def coefficients(step: int) -> tuple[float, float]:
    ramp = max(0.0, 1 - step / 100)
    return 0.01 ** ramp, 0.1 ** ramp


class GraphedTrainer:
    """Drop-in replacement for calling train.take_step every iteration."""

    def __init__(self, task, model, lr=0.01, warmup=3):
        self.task, self.model = task, model
        self.opt = torch.optim.Adam(model.weights_list, lr=lr, betas=(0.5, 0.9), capturable=True)
        self.c_size = torch.ones(())
        self.c_agg = torch.ones(())
        self.step_num = 0
        self.warmup = warmup
        self.graph = None
        self.out = None

    def _set_coefficients(self):
        cs, ca = coefficients(self.step_num)
        self.c_size.fill_(cs)
        self.c_agg.fill_(ca)

    def _eager_step(self):
        self.opt.zero_grad(set_to_none=True)
        out = loss_terms(self.task, self.model, self.c_size, self.c_agg)
        out[0].backward()
        self.opt.step()
        return out

    def step(self, logger=None):
        self._set_coefficients()
        if self.step_num < self.warmup:
            # Warm-up on a side stream, as required before capture.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                out = self._eager_step()
            torch.cuda.current_stream().wait_stream(s)
        else:
            if self.graph is None:
                self.graph = torch.cuda.CUDAGraph()
                self.opt.zero_grad(set_to_none=True)
                with torch.cuda.graph(self.graph):
                    self.out = loss_terms(self.task, self.model, self.c_size, self.c_agg)
                    self.out[0].backward()
                    self.opt.step()
            self.graph.replay()
            out = self.out
        if logger is not None:
            loss, logits, x_mask, y_mask, KL_amounts, KL_names, total_KL, rec = out
            logger.log(self.step_num, logits, x_mask, y_mask, KL_amounts, KL_names, total_KL, rec, loss)
        self.step_num += 1
