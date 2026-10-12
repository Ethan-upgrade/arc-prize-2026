# Paper draft — ARC Prize 2026 Paper Track

> Status: DRAFT v1 (2026-10-12). Only the final leaderboard number is pending.
> Every number below can be recomputed with `python scripts/reproduce_tables.py` (reads `results/`).
> Limit: 1,500 words (Kaggle Writeup). Author: Chengjun Cai.

---

## Title

**What a compression-based ARC solver is missing: symbolic hints, a faster CompressARC, and a negative result that points to output size**

(Short title for the gallery: *Symbolic hints for CompressARC: what helps and what doesn't*)

---

## Abstract (~120 words)

CompressARC solves ARC tasks with no pretraining by training a small network per task to compress it. We ask whether cheap symbolic program search can supply it with useful prior knowledge. We add learnable, zero-initialised gates that let the model use the outputs of partially-correct DSL programs as logit hints. Across two pre-registered A/B tests on 80 ARC-AGI-1 tasks, these whole-answer hints did **not** help: the gain equalled the variance between random seeds. A failure analysis explains why: the DSL proposes transformations CompressARC already finds on its own, while its real weaknesses are elsewhere, most clearly output size (0/11 on shrinking tasks). A size-only hint was correct on every task it fired on (15/15) yet solved none: getting the size right is necessary but far from sufficient. We also make CompressARC 3.7× faster per process on Kaggle's T4 with CUDA graphs, which raised our ARC-AGI-2 leaderboard score from 0% to **1.67%**.

---

## 1. Introduction (~150 words)

- ARC-AGI measures skill acquisition from a few examples; ARC-AGI-2 is unsolved (top Kaggle 2025 score: 24%).
- Two families: program search (precise, brittle) and learned models (flexible, data-hungry). CompressARC (Liao & Gu, 2025) is unusual: it learns per task from scratch, with no pretraining, by minimising description length.
- Natural question: can the two families be combined *cheaply*, by letting search tell the learner what it found?
- Contributions:
  1. A gated hint channel that lets CompressARC use symbolic program outputs at near-zero cost when they are useless.
  2. A careful negative result, with seed-noise controls and a pre-registered decision rule.
  3. A failure analysis showing *which* information the learner lacks, and a size-only hint built from it.
  4. An engineering result: CUDA-graph training (3.7× per process on T4) that makes ~570 steps/task fit Kaggle's 12-hour limit.

## 2. Prior work (~120 words)

- **CompressARC** (Liao & Gu, 2025): MDL, per-task VAE-style decoder, ~20% ARC-AGI-1 eval, ~4% ARC-AGI-2. We use it unmodified as the learner.
- **DSL program search** (Icecuber 2020; many since): exhaustive search over hand-written primitives; strong on ARC-AGI-1, near zero on ARC-AGI-2.
- **Hybrid / neuro-symbolic** (e.g. SOAR, Pourcel et al. 2025; Das et al. 2025): typically use an LLM to *write* programs. We go the other way: programs inform a tiny learner, with no LLM and no pretraining.
- **Test-time training** (ARChitects, NVARC): pretrained LLMs adapted per task. Far larger compute; orthogonal to our question.

## 3. Method (~350 words)

### 3.1 Partial program search
46 grid primitives (geometry, cropping, object selection, scaling, gravity, fills, half-grid logic), composed up to depth 2 with BFS. Unlike usual DSL solvers, we keep programs that are *partially* right: output shape matches every training target and pixel accuracy ≥ 50%. A learned colour remap is applied when it makes a program exact. Top-3 programs by accuracy become hints.

### 3.2 Gated hint channel
For hint *k* with output grids *H_k*, we add `g_k · onehot(H_k)` to CompressARC's output logits, where `g_k` is a learnable scalar initialised at 0. In MDL terms the model now encodes the *residual* between hint and target; a useless hint leaves `g_k ≈ 0` and costs almost nothing.

### 3.3 Size-only hint
~20 rules map an input grid to an output size (e.g. bounding box of the largest object, number of objects × 1, cell size between separator lines). A rule is kept only if it is exact on every training pair; a test size is used only if all kept rules agree. The hint changes only how the answer is *decoded* (the crop length), not training. That allows a perfectly paired comparison: one training run, decoded with and without the hint.

### 3.4 CUDA-graph training
CompressARC issues ~70k tiny kernels per step; on Kaggle's GPUs it is launch-bound (<3% of VRAM used). We capture forward + backward + Adam into one CUDA graph and replay it. Two capture blockers in the original code (a per-call scalar tensor, a per-step host→device mask copy) are patched at import time. Reusing one side stream during warm-up was essential: a fresh stream per step reserved 3× the needed memory.

## 4. Results (~400 words)

### 4.1 Reproduction
Unmodified CompressARC, 8 ARC-AGI-1 eval tasks the authors solved within 200 steps: **7/8** solved in 400 steps on a T4.

### 4.2 Speed
| | eager | CUDA graph |
|---|---|---|
| RTX 3060 laptop, 1 process | 2.7–3.9 s/step | 0.28–0.36 s/step |
| Kaggle T4, 4 processes (per process) | 3.6 s/step¹ | 1.0 s/step (3.7×) |
| Kaggle T4, same 8 tasks: steps/s, whole GPU | 0.56 (1 process) | 4.1 (4 processes) |
| Correctness (same 8 tasks) | 7/8 | 7/8 |

¹ Measured on the 40 round-1 A/B tasks, not the 8 tasks in the other rows. Eager mode could not run 4 processes on the 8-task set within our budget. The 3.7× figure is therefore approximate. On identical tasks, throughput per GPU rose 7.3× (eager could not usefully share the GPU, so it ran 1 process).

### 4.3 Whole-answer hints (negative)
| Round | Tasks | Base (seed 0) | Base (seed 1) | Hinted |
|---|---|---|---|---|
| 1 (exploratory) | 40 | 8.5 | — | 8.5 |
| 2 (pre-registered, fresh tasks) | 40 | 5 | 7 | 7 |

The gates *were* used (gate > 0.5 in 35/40 tasks in round 1 and 33/40 in round 2), but the solve rate did not change beyond seed noise. In round 1, hints that were just "copy the input" looked harmful; round 2 removed them and still found no gain.

### 4.4 Failure analysis
On 99 ARC-AGI-1 eval test outputs from our full submission run:
- correct 20% · near-miss (≥90% of cells right) 21% · far 27% · wrong palette 18% · **wrong size 12%**
- tasks whose output is smaller than the input: **0/11 solved**
- in near-misses, **72%** of wrong cells are cells the model left unchanged that should have changed

### 4.5 Size-only hint
Rules fire on 18 test outputs of ARC-AGI-1 eval and are right on **18/18**. Paired A/B (one training run per task, decoded with and without the hint, 500 steps, 15 tasks): **0 → 0 solved**, no difference at any checkpoint. The predicted size was right on all 15.

### 4.6 Leaderboard (ARC-AGI-2, private hidden set)
| Version | Score |
|---|---|
| DSL only | 0.00% |
| DSL exact + CUDA-graph CompressARC (~570 steps/task) | **1.67%** |
| final (more steps; submitted after 10-17) | **[TBD]** |

## 5. Why whole-answer hints fail (theory, ~200 words)

CompressARC's architecture already builds in the symmetries and operations a DSL hint would supply: equivariance to rotations and reflections, cumulative-max and shift layers for propagation, and per-colour channels. A DSL program that is ≥ 50% right on training pairs is almost always made of those same operations. Its output therefore lies close to a point the learner reaches on its own early in training, and the hint saves at most a few steps. Under MDL this is expected: a hint lowers description length only by the information it contains that the model's prior lacks.

The failure analysis points to where that missing information is. Output size is a *global, discrete* quantity. CompressARC can only express it through soft masks, and searches over it at decode time. Symbolic rules get it exactly right when they fire. **Prediction: hints help in proportion to how far their content is from the learner's inductive bias.** The size-only experiment tested this and gave a sharper answer than we expected: on tasks whose output shrinks, size is not the bottleneck. With the size fixed correctly, CompressARC still fails on content. Its weakness on these tasks is *selection*: deciding which part of the input to keep. A crop length cannot express that.

## 6. Conclusion (~80 words)

Cheap symbolic search is not a free lunch for a compression-based learner. Its usefulness depends on whether it tells the learner something new. We release code, all experiment logs and a CUDA-graph CompressARC that is 3.7× faster per process on Kaggle hardware.

---

### Word count (body, excl. tables): ~1,350 — within budget; leave room for the size-hint result.

### Figures to make
1. Cover image: an ARC task with the hint channel overlaid (required by Kaggle).
2. Failure-category bar chart.
3. Solve curves: base vs hinted vs seed-1 (round 2).

### Facts to double-check before submission
- Liao & Gu ARC-AGI-2 number (blog says ~4%).
- NVARC 2025 score (24.0% per arcprize.org).
- Exact count of DSL primitives (46 in `arcsolver/dsl.py`).
