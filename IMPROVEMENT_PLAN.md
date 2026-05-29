# NanoGPT Contest — Improvement Plan

**Goal:** Beat the current best of **24.20 PPL** (val.bin) on the CSE 251B contest (≤100M params, lowest PPL on hidden test set).
**Deadline:** May 31, 2026. **Budget:** ~$20 / ~65 GPU-hrs (RTX 4090 class).

---

## 1. Where we are

| Config | Params | Iters | PPL (val.bin) | Notes |
|---|---|---|---|---|
| baseline (vanilla GPT) | 50.9M | 10k | 41.72 | n_layer=8, n_head=8, n_embd=512 |
| + weight_decay | 50.9M | 10k | 41.50 | −0.22 |
| + SwiGLU | 51.1M | 10k | 41.20 | −0.52 |
| + RMSNorm | 50.9M | 10k | 41.06 | −0.66 |
| + WSD | 50.9M | 10k | 39.46 | −2.26 |
| + Muon | 50.9M | 10k | 35.98 | **−5.74 (biggest single win)** |
| + dropout | 50.9M | 10k | 43.99 | +2.27 (hurts at this scale) |
| RoPE (big model) | 98.8M | 10k | 32.05 | not size-comparable |
| **combined, small** | **50.9M** | **10k** | **33.90** | RMSNorm+RoPE+Muon+WSD, `out_combined` |
| **combined, big, 1 epoch** | **98.8M** | **100k** | **24.20** | **current best**, `out_combined_1epoch` |

**Two reference baselines to keep straight:**
- Small-config 10k reference = **33.90** (fast iteration loop)
- Big-config (98.8M) = **24.20** (the number that actually matters; no 10k big-config reference exists yet)

---

## 2. Issues found in the code (confirmed by reading `model_combined.py` / `train_combined.py`)

### Issue A — Muon applied to embeddings + LM head (HIGH value fix)
`configure_optimizers` puts **every** 2D tensor into Muon, including the tied `wte`/`lm_head` weight (50304×672). Newton-Schulz orthogonalization should NOT run on the embedding matrix.

**Fix:** Hybrid optimizer (standard modded-nanogpt recipe):
- **AdamW** for: token embedding + LM head + all 1D params (RMSNorm gains)
- **Muon** for: transformer block matrices only (attn `c_attn`/`c_proj`, MLP weights)
- Decouple LRs: Muon hidden-matrix LR can go much higher (~0.02–0.05) than the embedding AdamW LR (~1e-3 – 3e-3).

Since Muon was already the biggest lever (−5.74), fixing *how* it's applied is the highest-expected-value change.

### Issue B — SwiGLU dropped from combined model (LOW effort, free)
`model_combined.py` MLP is plain GELU 4×. SwiGLU only bought −0.5 at 10k but is param-matched (hidden=1792 keeps 98.85M) and stacks. Port from `model_swiglu.py` (`SwiGLU` class, lines 94–108).

### Issue C — WSD schedule degenerate in the 1-epoch run (MEDIUM)
The 1-epoch command passed `warmup_iters=2000 wsd_stable_iters=1500`. Since stable < warmup, the stable plateau is **negative** — it silently ran warmup→cosine-decay, NOT true WSD. The "long stable + short decay" benefit (worth −2.3) was never realized.

**Fix:** Make the schedule a real WSD (warmup → long stable plateau at peak LR → short linear/cosine decay to 0 over the last ~20%). Add an assertion/guard so stable_iters > warmup_iters.

### Issue D — minor
- `head_dim = 672/12 = 56` (not a multiple of 64; slightly off tensor-core sweet spot).
- Only 1 epoch trained — loss curve was likely still descending; more tokens = lower PPL.

---

## 3. Action plan (ranked by ROI)

### Tier 1 — high confidence, cheap (DO FIRST)
1. **Hybrid AdamW + Muon optimizer split** (Issue A) — separate LR args.
2. **Re-add param-matched SwiGLU** (Issue B), hidden=1792.
3. **Fix WSD schedule** (Issue C) — real stable phase + short decay to 0.

### Tier 2 — known speedrun wins, small effort
4. **QK-norm** (RMSNorm on q,k before attention) — stabilizes, enables higher LR.
5. **Tune Muon hidden-matrix LR** upward now that it's decoupled.

### Tier 3 — biggest absolute gain, costs compute
6. **Train longer / more tokens** (2nd epoch+). Past Chinchilla-optimal at 100M/10BT, but PPL keeps dropping with tokens. One more full run fits the budget.

---

## 4. Methodology — iterate at 10k first

**Yes, start at 10k.** Rules:
- Each ablation must use a **self-contained 10k WSD schedule** (warmup→stable→decay all inside 10k). Never compare iter-10000 of a 100k schedule — LR is still near peak there, so the number is meaningless.
- **Fast loop:** ablate on the small (50.9M) config → beat **33.90**.
- **Validate:** before the final run, do at least ONE 10k run at the **big** (98.8M) config (no big 10k reference exists yet).
- **Final:** spend remaining compute on ONE full run (1+ epochs) with the winning config.

**Caveats:**
- Cannot resume the 24.20 checkpoint — SwiGLU + optimizer split change model/optimizer shapes → fresh run. Keep 24.20 as fallback.
- 10k rankings usually transfer to 100k, but Muon-LR interactions don't always — reserve budget for the validating full run.

---

## 5. Concrete next steps

1. [x] Edit `model_combined.py`: hybrid optimizer in `configure_optimizers`; add SwiGLU. *(commit b39bd65)*
2. [x] Edit `train_combined.py`: WSD multiplier + `warmup<stable` guard; `--use_swiglu/--hybrid_opt/--muon_lr/--adamw_lr/--min_lr_frac` flags. *(commit b39bd65)*
3. [ ] Run small-config 10k Run A (SwiGLU + decay-to-0, single Muon); target < 33.90.
4. [ ] Run small-config hybrid sweep (Muon LR 0.02 / 0.035 / 0.05); beat Run A.
5. [ ] (optional) Big-config (98.8M) 10k validation of winning config.
6. [ ] Launch final full run (1+ epoch) with winner; evaluate on val.bin; submit if < 24.20.
7. [ ] (optional) Add QK-norm; re-run 10k.

> Record actual run results in **EXPERIMENTS.md**.

### New CLI flags (commit b39bd65)
| Flag | Default | Meaning |
|---|---|---|
| `--use_swiglu` | False | SwiGLU MLP (param-matched, hidden≈8/3·n_embd) |
| `--mlp_hidden_dim` | 0 | override SwiGLU hidden dim (0=auto) |
| `--hybrid_opt` | False | Muon(block matrices) + AdamW(embed/head+norms) |
| `--muon_lr` | 0.02 | Muon LR (hybrid only) |
| `--adamw_lr` | 2e-3 | AdamW LR (hybrid only) |
| `--min_lr_frac` | 0.0 | WSD decay endpoint as fraction of peak (0=zero) |

**Removed:** `--min_lr` (replaced by `--min_lr_frac`). Requires `warmup_iters < wsd_stable_iters`. Keep `--dtype=bfloat16` (hybrid optimizer needs the GradScaler disabled).

### Reference commands

Smoke-test on the pod before any run:
```bash
cd nanoGPT && python smoke_test_combined.py
```

Run A — corrected baseline (SwiGLU + decay-to-0 + valid stable), single Muon:
```bash
python -u train_combined.py --dataset=fineweb \
  --n_layer=8 --n_head=8 --n_embd=512 \
  --batch_size=12 --gradient_accumulation_steps=8 --block_size=1024 \
  --max_iters=10000 --warmup_iters=500 --wsd_stable_iters=8000 --min_lr_frac=0.0 \
  --use_swiglu=True --learning_rate=0.003 --weight_decay=0.1 --momentum=0.95 \
  --eval_interval=500 --eval_iters=200 --device=cuda --dtype=bfloat16 --compile=True \
  --out_dir=out/combined_A_swiglu
```

Runs B/C/D — hybrid optimizer, sweep Muon LR (AdamW fixed at 2e-3):
```bash
for MLR in 0.02 0.035 0.05; do
python -u train_combined.py --dataset=fineweb \
  --n_layer=8 --n_head=8 --n_embd=512 \
  --batch_size=12 --gradient_accumulation_steps=8 --block_size=1024 \
  --max_iters=10000 --warmup_iters=500 --wsd_stable_iters=8000 --min_lr_frac=0.0 \
  --use_swiglu=True --hybrid_opt=True --muon_lr=$MLR --adamw_lr=2e-3 \
  --weight_decay=0.1 --momentum=0.95 \
  --eval_interval=500 --eval_iters=200 --device=cuda --dtype=bfloat16 --compile=True \
  --out_dir=out/combined_hybrid_mlr$MLR
done
```

Final big-config full run (winner config; n_embd=672, ~98.8M):
```bash
python -u train_combined.py --dataset=fineweb \
  --n_layer=12 --n_head=12 --n_embd=672 \
  --batch_size=12 --gradient_accumulation_steps=8 --block_size=1024 \
  --max_iters=100240 --warmup_iters=2000 --wsd_stable_iters=82000 --min_lr_frac=0.0 \
  --use_swiglu=True --hybrid_opt=True --muon_lr=<WINNER> --adamw_lr=2e-3 \
  --weight_decay=0.1 --momentum=0.95 \
  --eval_interval=500 --eval_iters=200 --device=cuda --dtype=bfloat16 --compile=True \
  --out_dir=out/out_combined_v2_1epoch
```

Evaluate:
```bash
python ../cse251b-nanogpt-contest-public/evaluate.py --model_dir out/out_combined_v2_1epoch --data val.bin
```
