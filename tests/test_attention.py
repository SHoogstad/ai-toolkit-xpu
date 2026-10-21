"""xpu_attention: exact kernel accuracy and hybrid routing.

Includes the failure mode that broke v1.1.0: keys with a large shared
component (Krea 2's first block), where XPU's fused SDPA backward is far off.
"""
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import xpu_attention as xa

orig_sdpa = getattr(F, "_xpu_orig_sdpa", F.scaled_dot_product_attention)
DEV, BF16 = "xpu", torch.bfloat16
failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        failures.append(name)


def grads(fn, q, k, v, go):
    q, k, v = (t.detach().clone().requires_grad_() for t in (q, k, v))
    o = fn(q, k, v)
    o.backward(go.to(o.device, o.dtype))
    return [t.detach().double().cpu() for t in (o, q.grad, k.grad, v.grad)]


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def f64_ref(mask, rep):
    def fn(q, k, v):
        k2, v2 = k.double().repeat_interleave(rep, 1), v.double().repeat_interleave(rep, 1)
        s = q.double() @ k2.transpose(-1, -2) / q.shape[-1] ** 0.5
        if mask is not None:
            m = mask.cpu()
            s = s.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else s + m.double()
        return torch.softmax(s, -1).nan_to_num(0) @ v2
    return fn


torch.manual_seed(0)

# 1) exact kernel vs float64: GQA + padding mask with fully masked rows, additive mask
B, Hq, Hkv, L, D = 2, 8, 2, 70, 64
q, k, v = (torch.randn(B, h, L, D) for h in (Hq, Hkv, Hkv))
pad = torch.ones(B, L, dtype=torch.bool)
pad[1, -13:] = False
for tag, mask in (("bool padding mask", pad[:, None, :, None] & pad[:, None, None, :]),
                  ("additive per-head mask", torch.randn(1, Hq, L, L))):
    go = torch.randn(B, Hq, L, D)
    mine = grads(lambda a, b, c: xa.exact_attention(a, b, c, attn_mask=mask.to(DEV)),
                 q.to(DEV, BF16), k.to(DEV, BF16), v.to(DEV, BF16), go)
    ref = grads(f64_ref(mask, Hq // Hkv), q.to(BF16), k.to(BF16), v.to(BF16), go)
    err = max(rel(a, b) for a, b in zip(mine, ref))
    check(f"exact kernel vs float64, {tag}", err < 5e-3, f"max rel err {err:.2e}")

# 2) the v1.1.0 failure mode: large shared key component -> extreme logits
Hq, Hkv, L, D = 4, 1, 2384, 128
g = torch.Generator().manual_seed(1)
base = torch.randn(1, 1, 1, D, generator=g)
q = (torch.randn(1, Hq, L, D, generator=g) + 0.5 * base) * 32
k = (torch.randn(1, Hkv, L, D, generator=g) + 2.0 * base) * 32
v = torch.randn(1, Hkv, L, D, generator=g)
go = torch.randn(1, Hq, L, D, generator=g)
qb, kb, vb = (t.to(BF16) for t in (q, k, v))
ref = grads(f64_ref(None, Hq), qb, kb, vb, go)
allm = torch.ones(1, 1, L, L, dtype=torch.bool, device=DEV)
mine = grads(lambda a, b, c: xa.exact_attention(a, b, c, attn_mask=allm), qb.to(DEV), kb.to(DEV), vb.to(DEV), go)
check("exact kernel, extreme shared-key logits: dq vs float64", rel(mine[1], ref[1]) < 1e-2,
      f"dq rel err {rel(mine[1], ref[1]):.2e}")


def fused(a, b, c):
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        return orig_sdpa(a, b, c, enable_gqa=True)


fl = grads(fused, qb.to(DEV), kb.to(DEV), vb.to(DEV), go)
print(f"      (for reference: XPU fused kernel dq rel err on the same input {rel(fl[1], ref[1]):.2e})")

# 3) hybrid routing: which path each call takes, and accuracy vs MATH
calls = {"fused": 0, "exact": 0}
_exact = xa.exact_attention


def count_exact(*a, **kw):
    calls["exact"] += 1
    return _exact(*a, **kw)


def count_fused(*a, **kw):
    calls["fused"] += 1
    return orig_sdpa(*a, **kw)


xa.exact_attention = count_exact
B, Hq, Hkv, L, D, n = 1, 16, 4, 1280, 128, 1200
padv = torch.zeros(B, L, dtype=torch.bool, device=DEV)
padv[:, :n] = True
tail = padv[:, None, :, None] & padv[:, None, None, :]
odd = tail.clone()
odd[..., 5, 7] = False
for tag, amp, mask, want in (("tail padding, moderate", 1.0, tail, "fused"),
                             ("tail padding, extreme", 30.0, tail, "exact"),
                             ("non-tail mask", 1.0, odd, "exact")):
    q, k, v = (torch.randn(B, h, L, D, device=DEV, dtype=BF16) * s for h, s in ((Hq, amp), (Hkv, amp), (Hkv, 1)))
    go = torch.randn(B, Hq, L, D, device=DEV, dtype=BF16)
    calls.update(fused=0, exact=0)
    mine = grads(lambda a, b, c: xa.hybrid_attention(a, b, c, mask, None, True, count_fused, 2500.0), q, k, v, go)
    path = "fused" if calls["fused"] else "exact"

    def math(a, b, c):
        with sdpa_kernel([SDPBackend.MATH]):
            return orig_sdpa(a, b, c, attn_mask=mask, enable_gqa=True)

    ref = grads(math, q, k, v, go)
    rows = mask.any(-1)[0, 0].cpu()
    err = max(rel(a[:, :, rows] if i < 2 else a, b[:, :, rows] if i < 2 else b)
              for i, (a, b) in enumerate(zip(mine, ref)))
    zero_pad = bool((mine[0][:, :, ~rows] == 0).all()) if (~rows).any() else True
    check(f"hybrid: {tag} -> {want}", path == want and err < 1e-2 and zero_pad,
          f"path {path}, max rel err vs MATH {err:.2e}, padded rows zero {zero_pad}")
xa.exact_attention = _exact

print(f"\n{len(failures)} failure(s)" if failures else "\nall attention tests passed")
sys.exit(1 if failures else 0)
