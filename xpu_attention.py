"""Exact, memory-efficient attention for Intel XPU.

On XPU any attn_mask sends F.scaled_dot_product_attention to the MATH backend,
which materializes the full (heads x L x S) score matrix in fp32: several GB per
call at DiT sequence lengths, and ~6x slower than the fused kernel. The fused
kernel can't simply be used instead: its backward overflows on extreme logits
(Krea 2's first block has |q|,|k| ~600 -> inf gradients).

This does what the fused kernel does structurally (process query chunks,
recompute in backward from the saved log-sum-exp, never store the full score
matrix) with MATH's numerics: scores, softmax and the P@V product in fp32.
Supports bool or additive masks broadcastable to (B, Hq, L, S) and GQA.
"""
import math

import torch

# fp32 bytes per (chunk x S) score tile across all heads; bounds the transient
# memory of one chunk (the backward holds ~4 such tiles at once)
TILE_BYTES = 128 * 1024 * 1024


def _chunk(B, H, S):
    return max(16, min(4096, TILE_BYTES // max(1, B * H * S * 4)))


def _group_mask(mask, B, Hkv, G, L, S):
    """mask broadcastable to (B, Hq, L, S) -> (Bm, Hkv|1, G|1, L, S) view."""
    if mask is None:
        return None
    while mask.dim() < 4:
        mask = mask.unsqueeze(0)
    Bm, Hm = mask.shape[0], mask.shape[1]
    mask = mask.expand(Bm, Hm, L, S)
    if Hm == 1:
        return mask.unsqueeze(2)  # Bm,1,1,L,S
    return mask.reshape(Bm, Hkv, G, L, S)


def _centered(k):
    """Keys minus their mean over the sequence, in fp32. Softmax is invariant
    to a per-row constant, so this changes nothing mathematically but removes
    a shared logit offset (Krea 2 block 0: ~3e4) that otherwise costs fp32
    precision in the scores and dq. Must be done in fp32: centering bf16 keys
    and rounding them again injects error the large logits amplify."""
    kf = k.float()
    return kf - kf.mean(dim=-2, keepdim=True)


def _scores(qc, kT, scale, mc):
    s = torch.matmul(qc, kT).mul_(scale)
    if mc is not None:
        if mc.dtype == torch.bool:
            s = s.masked_fill(~mc, float("-inf"))
        else:
            s = s + mc.float()
    return s


class _ExactAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, scale):
        B, Hq, L, D = q.shape
        Hkv, S = k.shape[1], k.shape[2]
        G = Hq // Hkv
        qg = q.view(B, Hkv, G, L, D)
        kT = _centered(k).unsqueeze(2).transpose(-1, -2)  # B,Hkv,1,D,S
        vf = v.float().unsqueeze(2)  # B,Hkv,1,S,Dv
        mg = _group_mask(mask, B, Hkv, G, L, S)
        out = torch.empty((B, Hkv, G, L, v.shape[-1]), device=q.device, dtype=q.dtype)
        # row max and softmax normalizer are kept separately rather than as one
        # log-sum-exp: with logits ~3e4 an fp32 lse only resolves ~0.004, so
        # exp(s - lse) rebuilt in backward wouldn't sum to 1, and that residue
        # times the keys' large shared component dominates dq
        rmax = torch.empty((B, Hkv, G, L, 1), device=q.device, dtype=torch.float32)
        rsum = torch.empty((B, Hkv, G, L, 1), device=q.device, dtype=torch.float32)
        c = _chunk(B, Hq, S)
        for i in range(0, L, c):
            j = min(L, i + c)
            s = _scores(qg[..., i:j, :].float(), kT, scale, None if mg is None else mg[..., i:j, :])
            m = s.amax(-1, keepdim=True)
            m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))  # fully masked rows
            p = torch.exp(s - m)
            denom = p.sum(-1, keepdim=True)
            out[..., i:j, :] = (torch.matmul(p, vf) / denom.clamp_min(1e-30)).to(q.dtype)
            rmax[..., i:j, :] = m
            rsum[..., i:j, :] = denom
            del s, p, m, denom
        ctx.save_for_backward(q, k, v, rmax, rsum, mask)
        ctx.scale = scale
        return out.view(B, Hq, L, v.shape[-1])

    @staticmethod
    def backward(ctx, dout):
        q, k, v, rmax, rsum, mask = ctx.saved_tensors
        scale = ctx.scale
        B, Hq, L, D = q.shape
        Hkv, S = k.shape[1], k.shape[2]
        G = Hq // Hkv
        qg = q.view(B, Hkv, G, L, D)
        kf = _centered(k).unsqueeze(2)  # B,Hkv,1,S,D
        kT = kf.transpose(-1, -2)
        vT = v.float().unsqueeze(2).transpose(-1, -2)  # B,Hkv,1,Dv,S
        mg = _group_mask(mask, B, Hkv, G, L, S)
        do = dout.reshape(B, Hkv, G, L, -1)
        # delta_i = sum_j P_ij dP_ij is computed per chunk from fp32 P and dP,
        # NOT via the usual shortcut dO_i . O_i: with large logit offsets (Krea 2
        # block 0: 3e3..3e4) dS = P (dP - delta) is almost pure cancellation, and
        # the bf16 rounding of O swamps the true gradient (dq off by 6x; the
        # fused XPU kernel has the same problem)
        dq = torch.empty_like(qg)
        dk = torch.zeros((B, Hkv, S, D), device=q.device, dtype=torch.float32)
        dv = torch.zeros((B, Hkv, S, v.shape[-1]), device=q.device, dtype=torch.float32)
        c = _chunk(B, Hq, S)
        for i in range(0, L, c):
            j = min(L, i + c)
            qc = qg[..., i:j, :].float()
            s = _scores(qc, kT, scale, None if mg is None else mg[..., i:j, :])
            # exactly the forward's P (fully masked rows: exp(-inf) = 0 / clamp = 0)
            p = torch.exp(s - rmax[..., i:j, :]).div_(rsum[..., i:j, :].clamp_min(1e-30))
            del s
            doc = do[..., i:j, :].float()
            dv += torch.matmul(p.transpose(-1, -2), doc).sum(2)
            ds = torch.matmul(doc, vT)  # dP
            delta = (p * ds).sum(-1, keepdim=True)
            ds.sub_(delta).mul_(p).mul_(scale)
            del delta
            del p
            dq[..., i:j, :] = torch.matmul(ds, kf).to(q.dtype)
            dk += torch.matmul(ds.transpose(-1, -2), qc).sum(2)
            del ds, qc, doc
        # keys were centered (kc = k - mean_S k): chain rule through the mean
        dk = dk - dk.mean(dim=-2, keepdim=True)
        return dq.view(B, Hq, L, D), dk.to(k.dtype), dv.to(v.dtype), None, None


def exact_attention(q, k, v, attn_mask=None, scale=None):
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return _ExactAttention.apply(q, k, v, attn_mask, float(scale))


def _tail_padding_length(mask, L, S):
    """If a bool mask only hides a padded tail -- mask[b, :, i, j] ==
    valid[i] & valid[j] with valid = first n positions, same n for the whole
    batch -- return n, else None. One device->host sync."""
    if mask.dtype != torch.bool or L != S:
        return None
    m = mask
    while m.dim() < 4:
        m = m.unsqueeze(0)
    if m.shape[-1] != S or m.shape[-2] != L:
        return None
    row = m.any(dim=-1)  # B,H,L
    col = m.any(dim=-2)  # B,H,S
    n_col = col.sum(dim=-1)
    is_outer = torch.equal(m, row.unsqueeze(-1) & col.unsqueeze(-2))
    stats = torch.stack([n_col.min(), n_col.max(), row.sum(dim=-1).min(), row.sum(dim=-1).max()]).tolist()
    n = int(stats[0])
    if not is_outer or len(set(int(x) for x in stats)) != 1 or n == 0:
        return None
    if not bool(col[..., :n].all()):
        return None
    return n


def hybrid_attention(q, k, v, attn_mask, scale, enable_gqa, orig_sdpa, flash_max):
    """Fused kernel where it's accurate, exact kernel where it isn't.

    A tail-padding mask is replaced by slicing the padding off (exactly
    equivalent; padded query rows get 0, as MATH gives fully masked rows),
    which lets the call run unmasked. If max|q| * max|k| * sqrt(D) <=
    flash_max the fused kernel is accurate (bf16-level error; measured on
    Krea 2-like inputs with a shared key component) and is used; otherwise
    (Krea 2 block 0: ~4e6) the exact chunked kernel."""
    L, S = q.shape[2], k.shape[2]
    n = _tail_padding_length(attn_mask, L, S) if attn_mask is not None else S
    if n is not None:
        qs, ks, vs = q[:, :, :n], k[:, :, :n], v[:, :, :n]
        mags = torch.stack([qs.detach().abs().amax(), ks.detach().abs().amax()]).float().tolist()
        if mags[0] * mags[1] * math.sqrt(q.shape[-1]) <= flash_max:
            out = orig_sdpa(qs, ks, vs, scale=scale, enable_gqa=enable_gqa)
            if n < L:
                pad = out.new_zeros(out.shape[0], out.shape[1], L - n, out.shape[3])
                out = torch.cat([out, pad], dim=2)
            return out
    return exact_attention(q, k, v, attn_mask=attn_mask, scale=scale)


def supported(q, k, v, attn_mask, dropout_p, is_causal):
    """Masked, 4-D, no dropout/causal, and heads compatible with GQA."""
    if attn_mask is None or dropout_p != 0.0 or is_causal:
        return False
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return False
    if q.dtype not in (torch.bfloat16, torch.float16) or q.device.type != "xpu":
        return False
    Hq, Hkv = q.shape[1], k.shape[1]
    if Hkv == 0 or Hq % Hkv or k.shape[0] != q.shape[0]:
        return False
    m = attn_mask
    if m.dim() > 4:
        return False
    while m.dim() < 4:
        m = m.unsqueeze(0)
    return m.shape[0] in (1, q.shape[0]) and m.shape[1] in (1, Hq) \
        and m.shape[2] in (1, q.shape[2]) and m.shape[3] in (1, k.shape[2])
