# Symbolic hints for CompressARC (ARC Prize 2026)

Code, raw results and paper draft for an ARC Prize 2026 Paper Track entry (ARC-AGI-2 competition).

**Question.** [CompressARC](https://github.com/iliao2345/CompressARC) solves ARC tasks with no
pretraining by training a small network per task. Can cheap symbolic program search give it useful
prior knowledge?

**Answer, from three controlled experiments: not in the forms we tried.**

| Hint given to CompressARC | Result |
|---|---|
| Output of the best partially-correct DSL programs, as gated logit biases | Gain equal to seed noise (2 rounds, 80 ARC-AGI-1 tasks, pre-registered decision rule) |
| Output size only, from symbolic size rules | Size right on 15/15 tasks, 0 → 0 solved (paired: one training run decoded both ways) |

A failure analysis explains why: the DSL proposes what CompressARC already finds on its own, and on
the tasks where it fails, size is not the bottleneck.

**Also included:** a CUDA-graph training step for CompressARC, about 3.7× faster per process on a
Kaggle T4 with identical solve rate. On the ARC-AGI-2 leaderboard it raised our score from 0% to 1.67%.

## Layout

| Path | What |
|---|---|
| `arcsolver/dsl.py` | 46 grid primitives |
| `arcsolver/search.py` | Exact program search; `partial_search` for hint programs |
| `arcsolver/hinted.py` | `HintedCompressor`: CompressARC + gated hint channels |
| `arcsolver/size_hint.py` | Symbolic output-size rules and a size-hinted decoder |
| `arcsolver/fast.py` | CUDA-graph training step (`GraphedTrainer`) and a vectorised answer decoder |
| `arcsolver/submit_worker.py` | Per-process solver used by the Kaggle submission |
| `arcsolver/ab.py`, `ab_size.py` | A/B harnesses for the hint experiments |
| `scripts/kaggle_*.py` | Push experiments and the submission notebook to Kaggle |
| `scripts/failure_analysis.py`, `size_rules.py` | Analyses that need no GPU |
| `scripts/reproduce_tables.py` | **Recomputes every number in the paper from `results/`** |
| `results/` | Raw per-task results of every experiment in the paper |
| `tests/` | Equivalence test for the vectorised decoder |
| `paper/draft.md` | Paper draft |
| `WORKLOG.md` | Day-by-day log, including dead ends (in Chinese) |

## Reproduce

```bash
# numbers in the paper, from the saved raw results (seconds, no GPU)
python scripts/reproduce_tables.py

# setup for running anything else
git clone https://github.com/arcprize/ARC-AGI-2.git data_repo
git clone https://github.com/iliao2345/CompressARC.git third_party/CompressARC
pip install torch numpy matplotlib tqdm

# CUDA-graph vs original step, on one task (needs a CUDA GPU)
python -m arcsolver.bench_fast 4cd1b7b2 40

# the vectorised decoder gives identical answers to the original
python tests/test_best_slice_point.py
```

The CUDA-graph step patches two spots in CompressARC at import time; CompressARC's own files are not modified.

## License

Code in this repository: MIT-0 (see `LICENSE`). CompressARC is MIT-licensed by Isaac Liao and is used as an unmodified dependency.
