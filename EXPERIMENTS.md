# Experiment Log

Running log of training runs and val.bin PPL. Strategy lives in `IMPROVEMENT_PLAN.md`.
Eval: `python ../cse251b-nanogpt-contest-public/evaluate.py --model_dir <out_dir> --data val.bin`

**Current best to beat: 24.20 PPL** (big config, 1 epoch, `out_combined_1epoch`).

## Conventions
- **Judge by the CONTEST val.bin** (5,169,152 tokens, `cse251b-proj/contest/val.bin`) — this is the grade proxy and matches the 33.90/24.20 references. Never train on val.
- WARNING: the 100M FineWeb val (`data/fineweb/val.bin`, ~100M tokens) gives a *different, rosier* number because we train on FineWeb. Do NOT compare it to 33.90/24.20. Always re-eval on contest val.
- "small" = n_layer=8, n_head=8, n_embd=512 (~50.9M). "big" = n_layer=12, n_head=12, n_embd=672 (~98.8M).
- Record the **min val PPL** reached, the iter it occurred, and ms/iter (from the log, post-compile).
- Eval cmd: `cp model_combined.py out/<run>/model.py && python ../cse251b-nanogpt-contest-public/evaluate.py --model_dir out/<run> --data ../../cse251b-proj/contest/val.bin --device cuda`

## Reference baselines (prior work)

| Run | Model | Iters | Optim | Sched | val PPL | Notes |
|---|---|---|---|---|---|---|
| baseline | small | 10k | AdamW | cosine | 41.72 | vanilla GPT |
| + Muon | small | 10k | Muon | — | 35.98 | biggest single win |
| combined (small) | small | 10k | Muon(all) | WSD | 33.90 | RMSNorm+RoPE+Muon+WSD |
| combined (big) | big | 100k | Muon(all) | WSD* | **24.20** | *degenerate WSD; current best |

## v2 runs (commit b39bd65: SwiGLU + hybrid opt + fixed WSD)

All val PPL below = **contest val.bin** (5,169,152 tokens) unless noted. `wsd_stable` = end of stable plateau.

| ID | Model | Iters | SwiGLU | Optim (LRs) | wsd_stable | contest PPL | ms/iter | Status |
|---|---|---|---|---|---|---|---|---|
| A | small | 10k | yes | Muon-all (0.003) | 8000 | **34.91** | ~1080 | done (worse than 33.90) |
| B | small | 10k | yes | hybrid (muon 0.02 / adamw 2e-3) | 2000 | — | — | running |
| C | small | 10k | yes | hybrid (muon 0.035 / adamw 2e-3) | 2000 | — | — | queued |
| D | small | 10k | yes | hybrid (muon 0.05 / adamw 2e-3) | 2000 | — | — | queued |
| big-val | big | 10k | yes | hybrid (winner) | 2000 | — | — | optional |
| FINAL | big | 100k | yes | hybrid (winner) | ~20000 | — | — | pending |

### Notes / observations
- **Run A = 34.91 contest PPL (vs 33.90 old combined) → ~1 PPL WORSE.** Also scored 30.82 on the 100M FineWeb val (misleadingly better — ignore; train-distribution leak).
- Run A changed TWO things vs the 33.90 baseline: +SwiGLU (~−0.5 expected) AND schedule shape (short-stable→long-decay BECAME long-stable→short-decay-to-0). Net +1 worse ⇒ **the schedule change cost ~1.5 PPL.**
- **Lesson: at a fixed 10k budget, a LONG cosine decay beats long-stable+short-decay.** Long-stable WSD only pays off for very long runs / checkpoint branching. Reverting B+ to short stable (`wsd_stable=2000`, decay over 80%).
- Run A did NOT test the hybrid optimizer (highest-value change) — still single-Muon @0.003 orthogonalizing the embedding. The real lever starts at Run B.

## Decisions / learnings
- Hybrid optimizer: Muon over block matrices only; AdamW over tied embedding/LM-head + RMSNorm gains. (Old code orthogonalized the 50304×672 embedding — wrong and slow.)
- WSD is now a multiplier on each group's base LR, decays to `min_lr_frac × peak`; `warmup < stable` enforced.
- `--min_lr` removed → use `--min_lr_frac`. Keep `--dtype=bfloat16` (hybrid opt needs GradScaler disabled).
- Cannot resume the 24.20 checkpoint for these changes (arch/optimizer shapes differ) — fresh runs.
