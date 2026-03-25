"""
Parameter Golf: Byte-Level JEPA — Joint Embedding Predictive Architecture.
  - Byte-level input (vocab=260, no tokenizer) — true to JEPA principles
  - Patch-based: 8 bytes per patch, encoder operates on patches
  - JEPA encoder (10L, 768d) predicts next-patch representations (MSE loss)
  - Byte decoder (3L, 384d, local attention) produces byte probabilities (CE loss)
  - EMA target encoder for JEPA prediction targets
  - VICReg regularization to prevent representation collapse
  - Ternary weights {-1,0,1} for ~61M params in <16MB
  - Sliding window eval + PPM bigram blend
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import torch
try:
    import zstandard
except ImportError:
    zstandard = None
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_byte260")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 4096))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # JEPA model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 260))
    patch_size = int(os.environ.get("PATCH_SIZE", 8))
    enc_dim = int(os.environ.get("ENC_DIM", 768))
    enc_layers = int(os.environ.get("ENC_LAYERS", 10))
    enc_heads = int(os.environ.get("ENC_HEADS", 12))
    enc_kv_heads = int(os.environ.get("ENC_KV_HEADS", 6))
    enc_mlp_mult = int(os.environ.get("ENC_MLP_MULT", 3))
    dec_dim = int(os.environ.get("DEC_DIM", 384))
    dec_layers = int(os.environ.get("DEC_LAYERS", 3))
    dec_heads = int(os.environ.get("DEC_HEADS", 6))
    dec_kv_heads = int(os.environ.get("DEC_KV_HEADS", 3))
    dec_mlp_mult = int(os.environ.get("DEC_MLP_MULT", 3))
    dec_window = int(os.environ.get("DEC_WINDOW", 32))
    patch_embed_dim = int(os.environ.get("PATCH_EMBED_DIM", 128))
    pred_dim = int(os.environ.get("PRED_DIM", 512))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # JEPA loss weights.
    jepa_weight = float(os.environ.get("JEPA_WEIGHT", 1.0))
    var_weight = float(os.environ.get("VAR_WEIGHT", 0.04))
    cov_weight = float(os.environ.get("COV_WEIGHT", 0.01))

    # Optimizer.
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    embed_lr = float(os.environ.get("EMBED_LR", 0.03))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))

    # Ternary + EMA.
    ternary_start_frac = float(os.environ.get("TERNARY_START_FRAC", 0.3))
    ema_tau_start = float(os.environ.get("EMA_TAU_START", 0.996))
    ema_tau_end = float(os.environ.get("EMA_TAU_END", 0.9999))

    # Eval.
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    ppm_enabled = bool(int(os.environ.get("PPM_ENABLED", "0")))
    ppm_alpha = float(os.environ.get("PPM_ALPHA", 0.95))


# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum, backend_steps, nesterov = group["lr"], group["momentum"], group["backend_steps"], group["nesterov"]
            total = sum(p.numel() for p in params)
            updates = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "buf" not in state:
                        state["buf"] = torch.zeros_like(g)
                    buf = state["buf"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                p.add_(updates[curr:curr + p.numel()].view_as(p).to(p.dtype), alpha=-lr)
                curr += p.numel()


# -----------------------------
# TERNARY QUANTIZATION
# -----------------------------

CONTROL_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight", "byte_pos")

def ternary_ste(w: Tensor) -> Tensor:
    if w.ndim != 2:
        return w
    scale = w.detach().abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    threshold = 0.7 * scale
    t = torch.zeros_like(w)
    t[w.detach() > threshold] = 1.0
    t[w.detach() < -threshold] = -1.0
    return w + (t * scale - w).detach()


def quantize_ternary(state_dict):
    quantized, scales, dtypes = {}, {}, {}
    passthrough, pt_dtypes = {}, {}
    stats = {"param_count": 0, "quant_bytes": 0}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += t.numel()
        if not t.is_floating_point():
            passthrough[name] = t
            stats["quant_bytes"] += t.numel() * t.element_size()
            continue
        is_ctrl = any(p in name for p in CONTROL_PATTERNS)
        if is_ctrl or t.numel() <= 65536:
            if is_ctrl:
                passthrough[name] = t.float().contiguous()
            else:
                pt_dtypes[name] = str(t.dtype).removeprefix("torch.")
                passthrough[name] = t.to(torch.float16).contiguous()
            stats["quant_bytes"] += passthrough[name].numel() * passthrough[name].element_size()
            continue
        t32 = t.float()
        sc = t32.abs().mean(dim=1).clamp(min=1e-8)
        thr = 0.7 * sc[:, None]
        q = torch.zeros_like(t32, dtype=torch.int8)
        q[t32 > thr] = 1
        q[t32 < -thr] = -1
        quantized[name] = q.contiguous()
        scales[name] = sc.to(torch.float16).contiguous()
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["quant_bytes"] += q.numel() + sc.numel() * 2
    obj = {"__quant_format__": "ternary_v1", "quantized": quantized, "scales": scales,
           "dtypes": dtypes, "passthrough": passthrough}
    if pt_dtypes:
        obj["passthrough_orig_dtypes"] = pt_dtypes
    return obj, stats


def dequantize_ternary(obj):
    out = {}
    pt_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name].float()
        out[name] = (q.float() * s[:, None]).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        o = t.detach().cpu().contiguous()
        orig = pt_dtypes.get(name)
        if isinstance(orig, str):
            o = o.to(getattr(torch, orig)).contiguous()
        out[name] = o
    return out


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
        chunks = []
        rem = n
        while rem > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(rem, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            rem -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local = global_tokens // (self.world_size * grad_accum_steps)
        span = local + 1
        chunk = self.stream.take(span * self.world_size)
        start = self.rank * span
        loc = chunk[start:start + span].to(dtype=torch.int64)
        x = loc[:-1].reshape(-1, seq_len)
        y = loc[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# -----------------------------
# BYTE-LEVEL BPB EVAL
# -----------------------------

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, stride=0):
    """Byte-level BPB eval. For bytes: BPB = loss / ln(2) directly."""
    seq_len = args.train_seq_len
    total = val_tokens.numel() - 1
    use_sliding = 0 < stride < seq_len
    batch_seqs = max(1, args.val_batch_size // seq_len)
    if use_sliding:
        n_win = (total - seq_len) // stride + 1
        ws = (n_win * rank) // world_size
        we = (n_win * (rank + 1)) // world_size
    else:
        total_seqs = total // seq_len
        ws = (total_seqs * rank) // world_size
        we = (total_seqs * (rank + 1)) // world_size
        stride = seq_len

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for wb in range(ws, we, batch_seqs):
            wend = min(wb + batch_seqs, we)
            if use_sliding:
                x = torch.stack([val_tokens[w * stride:w * stride + seq_len] for w in range(wb, wend)])
                y = torch.stack([val_tokens[w * stride + 1:w * stride + seq_len + 1] for w in range(wb, wend)])
            else:
                rs = wb * seq_len
                re = wend * seq_len + 1
                loc = val_tokens[rs:re]
                x = loc[:-1].reshape(-1, seq_len)
                y = loc[1:].reshape(-1, seq_len)
            x = x.to(device=device, dtype=torch.int64, non_blocking=True)
            y = y.to(device=device, dtype=torch.int64, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                if use_sliding:
                    pt = model(x, y, per_token=True).detach()
                    scored = pt[:, -stride:]
                    loss_sum += scored.to(torch.float64).sum()
                    count += scored.numel()
                    # Byte-level: each token is 1 byte (except specials 0-3)
                    scored_y = y[:, -stride:]
                    byte_mask = (scored_y >= 4).to(torch.float64)
                    byte_count += byte_mask.sum()
                else:
                    bl = model(x, y).detach()
                    loss_sum += bl.to(torch.float64) * float(y.numel())
                    count += float(y.numel())
                    byte_mask = (y >= 4).to(torch.float64)
                    byte_count += byte_mask.sum()

    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss = loss_sum / count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = count.item() / byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# PPM BIGRAM MODEL
# -----------------------------

def build_ppm_predictions(tokens, vocab_size):
    tok = tokens.numpy().astype(np.int64) if isinstance(tokens, Tensor) else tokens.astype(np.int64)
    n = len(tok) - 1
    base_prob = np.float32(1.0 / vocab_size)
    prev_tok, next_tok = tok[:n], tok[1:n + 1]
    bigram_counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    np.add.at(bigram_counts, (prev_tok, next_tok), 1)
    row_totals = bigram_counts.sum(axis=1)
    row_distinct = (bigram_counts > 0).sum(axis=1).astype(np.float64)
    denom = np.maximum(row_totals + row_distinct, 1.0)
    smoothed = bigram_counts / denom[:, None] + (row_distinct / denom)[:, None] * base_prob
    probs = smoothed[prev_tok, next_tok].astype(np.float32)
    probs[row_totals[prev_tok] < 2] = base_prob
    return probs


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class TernaryLinear(nn.Linear):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.use_ternary = False
        self._zero_init = False
    def forward(self, x):
        w = self.weight.to(x.dtype)
        if self.use_ternary:
            w = ternary_ste(w)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache = (0, None, None)

    def forward(self, seq_len, device, dtype):
        if self._cache[0] != seq_len or self._cache[1] is None or self._cache[1].device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cache = (seq_len, freqs.cos()[None, None], freqs.sin()[None, None])
        return self._cache[1].to(dtype), self._cache[2].to(dtype)


def apply_rotary(x, cos, sin):
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = TernaryLinear(dim, dim, bias=False)
        self.c_k = TernaryLinear(dim, kv_dim, bias=False)
        self.c_v = TernaryLinear(dim, kv_dim, bias=False)
        self.proj = TernaryLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x):
        B, L, D = x.shape
        q = self.c_q(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(L, x.device, q.dtype)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        if self.num_kv_heads != self.num_heads:
            r = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(2).expand(B, self.num_kv_heads, r, L, self.head_dim).reshape(B, self.num_heads, L, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.num_kv_heads, r, L, self.head_dim).reshape(B, self.num_heads, L, self.head_dim)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, L, D))


class MLP(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        h = mult * dim
        self.fc = TernaryLinear(dim, h, bias=False)
        self.proj = TernaryLinear(h, dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), 0.5).square())


class Block(nn.Module):
    def __init__(self, dim, heads, kv_heads, mlp_mult, rope_base, qk_gain):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, heads, kv_heads, rope_base, qk_gain)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x, x0):
        mix = self.resid_mix.to(x.dtype)
        x = mix[0][None, None] * x + mix[1][None, None] * x0
        x = x + self.attn_scale.to(x.dtype)[None, None] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(x.dtype)[None, None] * self.mlp(self.mlp_norm(x))
        return x


# -----------------------------
# JEPA MODEL
# -----------------------------

class BytePatcher(nn.Module):
    def __init__(self, vocab_size, patch_embed_dim, patch_size, enc_dim):
        super().__init__()
        self.patch_size = patch_size
        self.byte_embed = nn.Embedding(vocab_size, patch_embed_dim)
        self.proj = TernaryLinear(patch_size * patch_embed_dim, enc_dim, bias=False)

    def forward(self, byte_ids):
        B, L = byte_ids.shape
        emb = self.byte_embed(byte_ids)  # (B, L, patch_embed_dim)
        emb = emb.reshape(B, L // self.patch_size, self.patch_size * emb.size(-1))
        return F.rms_norm(self.proj(emb), (self.proj.out_features,))


class JEPAPredictor(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            TernaryLinear(dim, hidden, bias=False), nn.GELU(),
            TernaryLinear(hidden, hidden, bias=False), nn.GELU(),
            TernaryLinear(hidden, dim, bias=False),
        )
    def forward(self, x):
        return self.net(x)


class ByteDecoder(nn.Module):
    def __init__(self, enc_dim, dec_dim, patch_size, vocab_size, num_layers, heads, kv_heads, mlp_mult, rope_base, qk_gain, softcap):
        super().__init__()
        self.patch_size = patch_size
        self.softcap = softcap
        self.upsample = TernaryLinear(enc_dim, patch_size * dec_dim, bias=False)
        self.byte_pos = nn.Parameter(torch.zeros(patch_size, dec_dim, dtype=torch.float32))
        self.dec_byte_embed = nn.Embedding(vocab_size, dec_dim)
        self.blocks = nn.ModuleList([
            Block(dec_dim, heads, kv_heads, mlp_mult, rope_base, qk_gain)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = TernaryLinear(dec_dim, vocab_size, bias=False)

    def forward(self, encoder_out, byte_ids):
        B, num_patches, _ = encoder_out.shape
        seq_len = num_patches * self.patch_size
        # Upsample patches to byte resolution.
        x = self.upsample(encoder_out)
        x = x.reshape(B, seq_len, -1)
        # Add within-patch positional embeddings.
        pos = self.byte_pos.to(x.dtype).unsqueeze(0).repeat(1, num_patches, 1)
        x = x + pos
        # Add byte input embeddings (teacher forcing).
        x = x + self.dec_byte_embed(byte_ids).to(x.dtype)
        x0 = x
        for block in self.blocks:
            x = block(x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        logits = self.lm_head(x)
        return self.softcap * torch.tanh(logits / self.softcap)


class JEPAModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        a = args
        self.patch_size = a.patch_size
        self.jepa_weight = a.jepa_weight
        self.var_weight = a.var_weight
        self.cov_weight = a.cov_weight

        self.patcher = BytePatcher(a.vocab_size, a.patch_embed_dim, a.patch_size, a.enc_dim)

        # Encoder with U-Net skips.
        self.enc_layers_count = a.enc_layers
        n_enc = a.enc_layers // 2
        n_dec = a.enc_layers - n_enc
        self.n_enc, self.n_dec = n_enc, n_dec
        self.skip_weights = nn.Parameter(torch.ones(min(n_enc, n_dec), a.enc_dim, dtype=torch.float32))
        self.encoder_blocks = nn.ModuleList([
            Block(a.enc_dim, a.enc_heads, a.enc_kv_heads, a.enc_mlp_mult, a.rope_base, a.qk_gain_init)
            for _ in range(a.enc_layers)
        ])
        self.enc_norm = RMSNorm()

        self.predictor = JEPAPredictor(a.enc_dim, a.pred_dim)
        self.decoder = ByteDecoder(
            a.enc_dim, a.dec_dim, a.patch_size, a.vocab_size, a.dec_layers,
            a.dec_heads, a.dec_kv_heads, a.dec_mlp_mult, a.rope_base, a.qk_gain_init, a.logit_softcap,
        )

    def enable_ternary(self):
        for m in self.modules():
            if isinstance(m, TernaryLinear):
                m.use_ternary = True

    def disable_ternary(self):
        for m in self.modules():
            if isinstance(m, TernaryLinear):
                m.use_ternary = False

    def encode(self, patch_emb):
        x = patch_emb
        x0 = x
        skips = []
        for i in range(self.n_enc):
            x = self.encoder_blocks[i](x, x0)
            skips.append(x)
        for i in range(self.n_dec):
            if skips:
                x = x + self.skip_weights[i].to(x.dtype)[None, None] * skips.pop()
            x = self.encoder_blocks[self.n_enc + i](x, x0)
        return self.enc_norm(x)

    def forward(self, byte_ids, target_byte_ids, per_token=False, return_logits=False,
                ema_encoder=None):
        B, L = byte_ids.shape
        patch_emb = self.patcher(byte_ids)
        encoder_out = self.encode(patch_emb)

        # Byte decoder: produce byte-level predictions.
        logits = self.decoder(encoder_out, byte_ids)  # (B*L, vocab)
        targets = target_byte_ids.reshape(-1)

        if return_logits:
            return logits.view(B, L, -1)
        if per_token:
            return F.cross_entropy(logits.float(), targets, reduction="none").view(B, L)

        ce_loss = F.cross_entropy(logits.float(), targets, reduction="mean")

        # JEPA loss: predict next-patch representation.
        jepa_loss = torch.tensor(0.0, device=byte_ids.device)
        var_loss = torch.tensor(0.0, device=byte_ids.device)
        cov_loss = torch.tensor(0.0, device=byte_ids.device)

        if self.jepa_weight > 0 and ema_encoder is not None:
            pred = self.predictor(encoder_out[:, :-1])
            with torch.no_grad():
                target_repr = ema_encoder(patch_emb)
                target_repr = F.layer_norm(target_repr, (target_repr.size(-1),))
            jepa_loss = F.mse_loss(pred, target_repr[:, 1:].detach())

            # VICReg regularization.
            if self.var_weight > 0:
                std = encoder_out.std(dim=0)
                var_loss = F.relu(1.0 - std).mean()
            if self.cov_weight > 0:
                centered = encoder_out.reshape(-1, encoder_out.size(-1))
                centered = centered - centered.mean(0)
                cov = (centered.T @ centered) / max(centered.size(0) - 1, 1)
                cov.fill_diagonal_(0)
                cov_loss = (cov ** 2).sum() / encoder_out.size(-1)

        total = ce_loss + self.jepa_weight * jepa_loss + self.var_weight * var_loss + self.cov_weight * cov_loss
        return total


def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if (p.ndim < 2 or any(pat in name for pat in CONTROL_PATTERNS)) and p.dtype != torch.float32:
                p.data = p.data.float()


# -----------------------------
# TRAINING
# -----------------------------

def main():
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    use_compile = sys.platform != "win32" and bool(int(os.environ.get("USE_COMPILE", "1")))
    if use_compile:
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(True); enable_math_sdp(True)

    logfile = None
    if master:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg, console=True):
        if not master: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)

    log0(code, console=False)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    log0(f"val_tokens:{val_tokens.numel()-1} byte_level:true vocab:{args.vocab_size}")

    # Build model.
    base_model = JEPAModel(args).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, TernaryLinear): m.float()
    restore_low_dim_params_to_fp32(base_model)

    # EMA target encoder (copy of encoder blocks + norm, no grad).
    ema_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()
                 if k.startswith("encoder_blocks.") or k.startswith("enc_norm.") or k.startswith("skip_weights")}

    def ema_encoder_fn(patch_emb):
        """Run the EMA encoder on patch embeddings."""
        x = patch_emb; x0 = x; skips = []
        # Reconstruct encoder forward from EMA state.
        enc_blocks = base_model.encoder_blocks
        for i in range(base_model.n_enc):
            x = enc_blocks[i](x, x0); skips.append(x)
        for i in range(base_model.n_dec):
            if skips:
                sw = ema_state.get(f"skip_weights", base_model.skip_weights)
                x = x + sw[i].to(x.dtype)[None, None] * skips.pop()
            x = enc_blocks[base_model.n_enc + i](x, x0)
        return base_model.enc_norm(x)

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True) if use_compile else base_model
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer setup.
    all_named = list(base_model.named_parameters())
    matrix_params = [p for n, p in all_named if p.ndim == 2 and not any(pat in n for pat in CONTROL_PATTERNS)]
    scalar_params = [p for n, p in all_named if p.ndim < 2 or any(pat in n for pat in CONTROL_PATTERNS)]
    embed_params = [p for n, p in all_named if "byte_embed" in n or "dec_byte_embed" in n]
    matrix_params = [p for p in matrix_params if not any(p is e for e in embed_params)]

    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for g in opt_muon.param_groups: g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.Adam([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
                                  betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_embed = torch.optim.Adam([{"params": embed_params, "lr": args.embed_lr, "base_lr": args.embed_lr}],
                                 betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [opt_muon, opt_scalar, opt_embed]

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} model_type:jepa_byte_ternary")
    log0(f"enc:{args.enc_layers}L/{args.enc_dim}d dec:{args.dec_layers}L/{args.dec_dim}d patch:{args.patch_size}")
    log0(f"world_size:{world_size} grad_accum:{grad_accum_steps} seq_len:{args.train_seq_len}")
    log0(f"seed:{args.seed}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all():
        for o in optimizers: o.zero_grad(set_to_none=True)

    max_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed):
        if args.warmdown_iters <= 0: return 1.0
        if max_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if ws <= step < args.iterations else 1.0
        sms = elapsed / max(step, 1)
        wms = args.warmdown_iters * sms
        rem = max(max_ms - elapsed, 0.0)
        return rem / max(wms, 1e-9) if rem <= wms else 1.0

    ternary_active = False
    ternary_start_ms = max_ms * args.ternary_start_frac if max_ms and args.ternary_start_frac < 1.0 else float("inf")

    # Warmup.
    if args.warmup_steps > 0:
        init_s = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
        init_o = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                (loss * grad_scale).backward()
            for o in optimizers: o.step()
            zero_grad_all()
            if ws + 1 == args.warmup_steps or (ws + 1) % 10 == 0:
                log0(f"warmup_step:{ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(init_s, strict=True)
        for o, s in zip(optimizers, init_o): o.load_state_dict(s)
        zero_grad_all()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Main loop.
    train_ms = 0.0
    stop_at = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last = step == args.iterations or (stop_at is not None and step >= stop_at)
        if last or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            train_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} "
                 f"train_time:{train_ms:.0f}ms step_avg:{train_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last: break

        elapsed = train_ms + 1000.0 * (time.perf_counter() - t0)
        if not ternary_active and elapsed >= ternary_start_ms:
            base_model.enable_ternary()
            ternary_active = True
            log0(f"ternary:activated step:{step} elapsed:{elapsed:.0f}ms")

        scale = lr_mul(step, elapsed)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for ms in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y, ema_encoder=ema_encoder_fn)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        mm = (1-frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in opt_muon.param_groups: g["momentum"] = mm
        for o in optimizers:
            for g in o.param_groups: g["lr"] = g["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for o in optimizers: o.step()
        zero_grad_all()

        # EMA update.
        tau_frac = min(step / max(args.iterations, 1), 1.0)
        tau = args.ema_tau_start + (args.ema_tau_end - args.ema_tau_start) * tau_frac
        with torch.no_grad():
            for k, v in base_model.state_dict().items():
                if k in ema_state:
                    ema_state[k].lerp_(v, 1.0 - tau)

        step += 1
        approx = train_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx:.0f}ms step_avg:{approx/step:.2f}ms")

        reached = max_ms is not None and approx >= max_ms
        if distributed and max_ms:
            cap = torch.tensor(int(reached), device=device)
            dist.all_reduce(cap, op=dist.ReduceOp.MAX)
            reached = bool(cap.item())
        if stop_at is None and reached:
            stop_at = step

    log0(f"peak memory: {torch.cuda.max_memory_allocated()//1024//1024} MiB")

    if ternary_active:
        base_model.disable_ternary()

    # Serialize.
    qobj, qstats = quantize_ternary(base_model.state_dict())
    buf = io.BytesIO(); torch.save(qobj, buf); raw = buf.getvalue()
    if zstandard:
        blob = zstandard.ZstdCompressor(level=22).compress(raw); cn = "zstd-22"
    else:
        blob = zlib.compress(raw, level=9); cn = "zlib-9"
    if master:
        with open("final_model.ptz", "wb") as f: f.write(blob)
        qb = os.path.getsize("final_model.ptz")
        cb = len(code.encode("utf-8"))
        log0(f"Serialized {cn}: {qb} bytes code:{cb} total:{qb+cb}")

    # Roundtrip eval.
    if distributed: dist.barrier()
    with open("final_model.ptz", "rb") as f: blob_disk = f.read()
    raw_disk = zstandard.ZstdDecompressor().decompress(blob_disk) if zstandard else zlib.decompress(blob_disk)
    qs = torch.load(io.BytesIO(raw_disk), map_location="cpu")
    base_model.load_state_dict(dequantize_ternary(qs), strict=True)

    torch.cuda.synchronize(); te = time.perf_counter()
    ql, qb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens)
    log0(f"final_roundtrip val_loss:{ql:.4f} val_bpb:{qb:.4f} eval:{1000*(time.perf_counter()-te):.0f}ms")
    log0(f"final_roundtrip_exact val_loss:{ql:.8f} val_bpb:{qb:.8f}")

    # Sliding window.
    if args.eval_stride > 0 and args.eval_stride < args.train_seq_len:
        torch.cuda.synchronize(); ts = time.perf_counter()
        sl, sb = eval_val(args, base_model, rank, world_size, device, grad_accum_steps, val_tokens, stride=args.eval_stride)
        log0(f"final_sliding val_loss:{sl:.4f} val_bpb:{sb:.4f} stride:{args.eval_stride} eval:{1000*(time.perf_counter()-ts):.0f}ms")
        log0(f"final_sliding_exact val_loss:{sl:.8f} val_bpb:{sb:.8f}")

    if distributed: dist.destroy_process_group()


if __name__ == "__main__":
    main()
