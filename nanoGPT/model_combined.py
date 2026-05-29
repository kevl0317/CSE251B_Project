"""
GPT model combining three modifications:
  - RMSNorm (instead of LayerNorm)
  - RoPE rotary position embeddings (instead of learned absolute wpe)
  - Muon optimizer (momentum + Newton-Schulz orthogonalization)
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# Muon optimizer
# -----------------------------------------------------------------------------

def zeropower_via_newtonschulz(G, steps=5):
    """Orthogonalize a matrix via Newton-Schulz iteration."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X /= (X.norm() + 1e-7)
    if X.size(0) > X.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon: momentum + Newton-Schulz for 2D params, momentum SGD for 1D."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, wd=0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, wd=wd)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            wd = group['wd']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if wd > 0:
                    p.mul_(1 - lr * wd)
                state = self.state[p]
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(g)
                buf = state['buf']
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                if g.ndim == 2 and g.size(0) > 1 and g.size(1) > 1:
                    update = zeropower_via_newtonschulz(g, steps=ns_steps)
                    scale = max(g.size(-2), g.size(-1)) ** 0.5
                    p.add_(update, alpha=-lr * scale * 0.2)
                else:
                    p.add_(g, alpha=-lr)


# -----------------------------------------------------------------------------
# CombinedOptimizer: drive Muon + AdamW together (hybrid recipe)
# -----------------------------------------------------------------------------

class CombinedOptimizer:
    """Thin wrapper that steps several sub-optimizers as one.

    Exposes a concatenated `param_groups` (referencing the real group dicts) so the
    training loop can set per-group LRs uniformly, while each group keeps its own
    base LR via 'initial_lr'. Only the methods the training loop uses are provided.
    """

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)
        self.param_groups = [g for opt in self.optimizers for g in opt.param_groups]

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, *args, **kwargs):
        for opt in self.optimizers:
            opt.step()

    def state_dict(self):
        return {'combined': [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self.optimizers, state_dict['combined']):
            opt.load_state_dict(sd)


# -----------------------------------------------------------------------------
# RMSNorm
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, input):
        x = input.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.weight).type_as(input)


# -----------------------------------------------------------------------------
# RoPE helpers
# -----------------------------------------------------------------------------

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# -----------------------------------------------------------------------------
# Transformer components
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        head_dim = config.n_embd // config.n_head

        # RoPE frequency embeddings (precomputed up to block_size)
        theta = 10000.0
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(config.block_size)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Apply RoPE to query and key
        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, hs)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_emb(q, k, cos, sin)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SwiGLU(nn.Module):
    # LLaMA-style SwiGLU MLP: down(silu(gate(x)) * up(x)).
    # Hidden dim defaults to ~(8/3)*n_embd to match the vanilla MLP's 8*d^2 param
    # budget; set config.mlp_hidden_dim explicitly to override. Down projection is
    # named c_proj so the residual scaled-init in GPT.__init__ still applies.
    def __init__(self, config):
        super().__init__()
        hidden = config.mlp_hidden_dim if config.mlp_hidden_dim else int(round(8 * config.n_embd / 3))
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.up_proj   = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.c_proj    = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, bias=config.bias)
        self.mlp = SwiGLU(config) if config.use_swiglu else MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# -----------------------------------------------------------------------------
# GPT model
# -----------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    use_swiglu: bool = False   # replace GELU MLP with LLaMA-style SwiGLU
    mlp_hidden_dim: int = 0    # SwiGLU hidden dim; 0 = auto (~8/3 * n_embd, param-matched)


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=False):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x)
            return logits

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
            # Update RoPE buffers for new block size
            head_dim = self.config.n_embd // self.config.n_head
            theta = 10000.0
            inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
            t = torch.arange(block_size)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            block.attn.cos = emb.cos()
            block.attn.sin = emb.sin()

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """Load pretrained GPT-2 weights (skips wpe since we use RoPE)."""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.wpe.weight')]
        sd_keys = [k for k in sd_keys if not k.endswith('.cos') and not k.endswith('.sin')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), \
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    # -------------------------------------------------------------------------
    # Muon optimizer
    # -------------------------------------------------------------------------

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type,
                             hybrid=False, muon_lr=0.02, adamw_lr=2e-3, adamw_betas=(0.9, 0.95)):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        momentum = betas[0] if isinstance(betas, tuple) else betas

        if not hybrid:
            # Original recipe: a single Muon optimizer over every parameter.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            optim_groups = [
                {'params': decay_params, 'lr': learning_rate, 'momentum': momentum,
                 'nesterov': True, 'ns_steps': 5, 'wd': weight_decay},
                {'params': nodecay_params, 'lr': learning_rate, 'momentum': momentum,
                 'nesterov': True, 'ns_steps': 5, 'wd': 0.0},
            ]
            optimizer = Muon(optim_groups, lr=learning_rate, momentum=momentum)
            print(f"using Muon optimizer with momentum={momentum}")
            return optimizer

        # Hybrid recipe: Muon over the transformer block matrices only; AdamW over
        # the (tied) token embedding / LM head and all 1D params (RMSNorm gains).
        muon_params, adamw_decay, adamw_nodecay = [], [], []
        for n, p in param_dict.items():
            if p.dim() < 2:
                adamw_nodecay.append(p)                      # RMSNorm gains
            elif 'wte' in n or 'lm_head' in n:
                adamw_decay.append(p)                        # tied embedding / output head
            else:
                muon_params.append(p)                        # attn + MLP block matrices
        n_muon = sum(p.numel() for p in muon_params)
        n_adamw = sum(p.numel() for p in adamw_decay + adamw_nodecay)
        print(f"hybrid optimizer: Muon over {len(muon_params)} matrices "
              f"({n_muon:,} params, lr={muon_lr}); AdamW over "
              f"{len(adamw_decay) + len(adamw_nodecay)} tensors ({n_adamw:,} params, lr={adamw_lr})")

        muon = Muon(
            [{'params': muon_params, 'lr': muon_lr, 'momentum': momentum,
              'nesterov': True, 'ns_steps': 5, 'wd': weight_decay}],
            lr=muon_lr, momentum=momentum)
        use_fused = (device_type == 'cuda')
        adamw = torch.optim.AdamW(
            [{'params': adamw_decay, 'weight_decay': weight_decay},
             {'params': adamw_nodecay, 'weight_decay': 0.0}],
            lr=adamw_lr, betas=adamw_betas, fused=use_fused)
        return CombinedOptimizer([muon, adamw])

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    if model.config.vocab_size != 50257:
        model = GPTEvalWrapper(model, model.config.vocab_size, 50257)
    return model


class GPTEvalWrapper(torch.nn.Module):

    def __init__(self, model, original_vocab_size, target_vocab_size):
        super().__init__()
        self.model = model
        self.original_vocab_size = original_vocab_size
        self.target_vocab_size = target_vocab_size

    def forward(self, idx):
        logits = self.model(idx)
        if self.original_vocab_size > self.target_vocab_size:
            logits = logits[..., :self.target_vocab_size]
        elif self.original_vocab_size < self.target_vocab_size:
            padding = torch.zeros(
                logits.shape[:-1] + (self.target_vocab_size - self.original_vocab_size,),
                device=logits.device, dtype=logits.dtype)
            logits = torch.cat([logits, padding], dim=-1)
        return logits

    def to(self, device):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self
