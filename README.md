# Symbolic hints for CompressARC (ARC Prize 2026)

Work in progress for the ARC Prize 2026 Paper Track (ARC-AGI-2).

**Idea.** A cheap DSL program search finds programs that are *partially* right on the
training pairs. Their outputs are fed to [CompressARC](https://github.com/iliao2345/CompressARC)
as additive logit hints, each with a learnable gate initialised at zero. The per-task model
then only has to encode the residual between the hint and the target; a useless hint costs
almost nothing because its gate stays near zero.

## Layout

| Path | What |
|---|---|
| `arcsolver/dsl.py` | Grid primitives |
| `arcsolver/search.py` | Exact program search, and `partial_search` for hint programs |
| `arcsolver/evaluate.py` | Local evaluator (official rule: 2 attempts, exact match) |
| `arcsolver/hinted.py` | `HintedCompressor`: CompressARC + gated hint channels |
| `arcsolver/ab.py` | A/B harness: CompressARC vs. CompressARC + hints |
| `scripts/kaggle_*.py` | Run experiments / submissions on Kaggle |
| `WORKLOG.md` | Progress log and results (in Chinese) |

## Setup

```bash
git clone https://github.com/arcprize/ARC-AGI-2.git data_repo
git clone https://github.com/iliao2345/CompressARC.git third_party/CompressARC
pip install torch numpy matplotlib tqdm
python -m arcsolver.evaluate --data data_repo/data/evaluation
```

## License

Code in this repository: MIT-0 (see `LICENSE`).
CompressARC is MIT-licensed by Isaac Liao and is used unmodified as a dependency.
