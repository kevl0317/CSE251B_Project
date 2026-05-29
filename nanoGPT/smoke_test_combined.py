"""
Quick correctness check for the SwiGLU + hybrid-optimizer + WSD changes.
Run on the pod (needs torch) BEFORE launching a real run:

    python smoke_test_combined.py

Exits non-zero on any failure. Does NOT touch data or write checkpoints.
"""
import torch
from model_combined import GPTConfig, GPT, CombinedOptimizer, Muon

fails = []
def check(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        fails.append(name)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1) Big contest config + SwiGLU stays <= 100M params
big = GPT(GPTConfig(block_size=1024, vocab_size=50304, n_layer=12, n_head=12,
                    n_embd=672, bias=False, use_swiglu=True))
n = sum(p.numel() for p in big.parameters())
check("big SwiGLU config <= 100M", n <= 100_000_000, f"-> {n:,} params")
check("SwiGLU MLP active", type(big.transformer.h[0].mlp).__name__ == "SwiGLU")

# 2) Hybrid optimizer split is complete and correct
opt = big.configure_optimizers(0.1, 0.003, (0.95, 0.0), dev,
                               hybrid=True, muon_lr=0.02, adamw_lr=2e-3)
check("hybrid returns CombinedOptimizer", type(opt).__name__ == "CombinedOptimizer")
muon, adamw = opt.optimizers
muon_ids = {id(p) for grp in muon.param_groups for p in grp['params']}
adamw_ids = {id(p) for grp in adamw.param_groups for p in grp['params']}
all_ids = [id(p) for p in big.parameters() if p.requires_grad]
covered = muon_ids | adamw_ids
check("no param dropped/duplicated", len(muon_ids & adamw_ids) == 0 and covered == set(all_ids))
check("embedding/head in AdamW (not Muon)",
      id(big.transformer.wte.weight) in adamw_ids and id(big.transformer.wte.weight) not in muon_ids)
check("attn matrix in Muon", id(big.transformer.h[0].attn.c_attn.weight) in muon_ids)
check("RMSNorm gain in AdamW", id(big.transformer.ln_f.weight) in adamw_ids)

# 3) Non-hybrid path still returns plain Muon
check("non-hybrid returns Muon",
      type(big.configure_optimizers(0.1, 0.003, (0.95, 0.0), dev, hybrid=False)).__name__ == "Muon")

# 4) One full train step on a tiny model (forward/backward/step/state_dict)
tiny = GPT(GPTConfig(block_size=64, vocab_size=512, n_layer=2, n_head=4,
                     n_embd=64, bias=False, use_swiglu=True)).to(dev)
o = tiny.configure_optimizers(0.1, 0.003, (0.95, 0.0), dev, hybrid=True)
for g in o.param_groups:
    g.setdefault('initial_lr', g['lr'])
x = torch.randint(0, 512, (2, 64), device=dev)
y = torch.randint(0, 512, (2, 64), device=dev)
_, loss = tiny(x, y)
loss.backward()
torch.nn.utils.clip_grad_norm_(tiny.parameters(), 1.0)
for g in o.param_groups:                 # simulate a schedule multiplier
    g['lr'] = g['initial_lr'] * 0.5
o.step()
o.zero_grad(set_to_none=True)
o.load_state_dict(o.state_dict())        # round-trip
check("hybrid train step + state_dict round-trip", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")

print()
if fails:
    print("FAILURES:", fails)
    raise SystemExit(1)
print("ALL CHECKS PASSED")
