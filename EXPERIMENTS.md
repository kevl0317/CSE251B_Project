# Experiment Log

Running log of training runs and val.bin PPL. Strategy lives in `IMPROVEMENT_PLAN.md`.
Eval: `python ../cse251b-nanogpt-contest-public/evaluate.py --model_dir <out_dir> --data val.bin`

**Current best to beat: 24.20 PPL** (big config, 1 epoch, `out_combined_1epoch`).

## Conventions
- All PPL on the public `val.bin` (~5.17M tokens). Never train on val.
- "small" = n_layer=8, n_head=8, n_embd=512 (~50.9M). "big" = n_layer=12, n_head=12, n_embd=672 (~98.8M).
- Record the **min val PPL** reached, the iter it occurred, and ms/iter (from the log, post-compile).

## Reference baselines (prior work)

| Run | Model | Iters | Optim | Sched | val PPL | Notes |
|---|---|---|---|---|---|---|
| baseline | small | 10k | AdamW | cosine | 41.72 | vanilla GPT |
| + Muon | small | 10k | Muon | — | 35.98 | biggest single win |
| combined (small) | small | 10k | Muon(all) | WSD | 33.90 | RMSNorm+RoPE+Muon+WSD |
| combined (big) | big | 100k | Muon(all) | WSD* | **24.20** | *degenerate WSD; current best |

## v2 runs (commit b39bd65: SwiGLU + hybrid opt + fixed WSD)

| ID | Model | Iters | SwiGLU | Optim (LRs) | min_lr_frac | val PPL | best iter | ms/iter | Status |
|---|---|---|---|---|---|---|---|---|---|
| A | small | 10k | yes | Muon-all (0.003) | 0.0 | — | — | — | queued |
| B | small | 10k | yes | hybrid (muon 0.02 / adamw 2e-3) | 0.0 | — | — | — | queued |
| C | small | 10k | yes | hybrid (muon 0.035 / adamw 2e-3) | 0.0 | — | — | — | queued |
| D | small | 10k | yes | hybrid (muon 0.05 / adamw 2e-3) | 0.0 | — | — | — | queued |
| big-val | big | 10k | yes | hybrid (winner) | 0.0 | — | — | — | optional |
| FINAL | big | 100k | yes | hybrid (winner) | 0.0 | — | — | — | pending |

### Notes / observations
- (fill in as runs complete — e.g. which Muon LR won, any instability, loss-curve shape)

## Decisions / learnings
- Hybrid optimizer: Muon over block matrices only; AdamW over tied embedding/LM-head + RMSNorm gains. (Old code orthogonalized the 50304×672 embedding — wrong and slow.)
- WSD is now a multiplier on each group's base LR, decays to `min_lr_frac × peak`; `warmup < stable` enforced.
- `--min_lr` removed → use `--min_lr_frac`. Keep `--dtype=bfloat16` (hybrid opt needs GradScaler disabled).
- Cannot resume the 24.20 checkpoint for these changes (arch/optimizer shapes differ) — fresh runs.
