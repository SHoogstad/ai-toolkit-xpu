"""Correctness, speed and memory of xpu_attention.exact_attention vs SDPA MATH.

usage (in the container): python3 xpu_attention_test.py [/path/to/xpu_attention.py dir]
"""
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.12/dist-packages")
from xpu_attention import exact_attention  # noqa: E402

orig_sdpa = getattr(F, "_xpu_orig_sdpa", F.scaled_dot_product_attention)


def grads(fn, q, k, v, go):
    q, k, v = (t.detach().clone().requires_grad_() for t in (q, k, v))
    o = fn(q, k, v)
    o.backward(go)
    return [t.float() for t in (o, q.grad, k.grad, v.grad)]


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def compare(tag, q, k, v, mask, gqa, ref="math"):
    go = torch.randn(q.shape[0], q.shape[1], q.shape[2], v.shape[-1], device=q.device, dtype=q.dtype)
    mine = grads(lambda a, b, c: exact_attention(a, b, c, attn_mask=mask), q, k, v, go)
    if ref == "f64":
        def refn(a, b, c):
            rep = a.shape[1] // b.shape[1]
            b2, c2 = b.double().repeat_interleave(rep, 1), c.double().repeat_interleave(rep, 1)
            s = a.double() @ b2.transpose(-1, -2) / a.shape[-1] ** 0.5
            if mask is not None:
                mc = mask.cpu()
                s = s.masked_fill(~mc, float("-inf")) if mc.dtype == torch.bool else s + mc.double()
            return (torch.softmax(s, -1).nan_to_num(0) @ c2).to(a.dtype)
        theirs = grads(refn, q.cpu(), k.cpu(), v.cpu(), go.cpu())
        mine = [t.cpu() for t in mine]
    else:
        def refn(a, b, c):
            with sdpa_kernel([SDPBackend.MATH]):
                return orig_sdpa(a, b, c, attn_mask=mask, enable_gqa=gqa)
        theirs = grads(refn, q, k, v, go)
    fin = all(bool(torch.isfinite(t).all()) for t in mine)
    errs = ", ".join(f"{n} {rel(a, b):.2e}" for n, a, b in zip(("out", "dq", "dk", "dv"), mine, theirs))
    print(f"{tag:46s} finite {fin} | rel err vs {ref}: {errs}", flush=True)
    return fin and max(rel(a, b) for a, b in zip(mine, theirs)) < 2e-2


ok = True
dev, dt = "xpu", torch.bfloat16
torch.manual_seed(0)
# 1) small case vs float64 CPU reference, with padding mask and fully-masked rows
B, Hq, Hkv, L, D = 2, 8, 2, 70, 64
q, k, v = (torch.randn(B, h, L, D, device=dev, dtype=dt) for h in (Hq, Hkv, Hkv))
pad = torch.ones(B, L, dtype=torch.bool, device=dev)
pad[1, -13:] = False
mask = pad[:, None, :, None] & pad[:, None, None, :]
ok &= compare("small GQA + padding mask (vs float64)", q, k, v, mask, True, ref="f64")
addm = torch.randn(1, Hq, L, L, device=dev, dtype=dt)
ok &= compare("small GQA + additive per-head mask (vs float64)", q, k, v, addm, True, ref="f64")

# 2) Krea 2 main-block shapes vs MATH
B, Hq, Hkv, L, D = 1, 48, 12, 2384, 128
q, k, v = (torch.randn(B, h, L, D, device=dev, dtype=dt) for h in (Hq, Hkv, Hkv))
allm = torch.ones(B, 1, L, L, dtype=torch.bool, device=dev)
ok &= compare("Krea2 shape, all-true mask (vs MATH)", q, k, v, allm, True)
q8, k8 = q * 50, k * 50  # extreme logits like block 0
ok &= compare("Krea2 shape, extreme logits x50 (vs MATH)", q8, k8, v, allm, True)

# 3) real captured block-0 tensors, if available
cap = "/app/ai-toolkit/output/attn_capture.pt"
if os.path.exists(cap):
    c0 = torch.load(cap)[0]
    qc, kc, vc = (c0[n].to(dev) for n in ("q", "k", "v"))
    m0 = torch.ones(1, 1, qc.shape[2], kc.shape[2], dtype=torch.bool, device=dev)
    ok &= compare(f"captured Krea2 block 0 |q|max {qc.float().abs().max():.0f} (vs MATH)", qc, kc, vc, m0, True)

# 4) speed + peak memory, fwd+bwd, Krea 2 shape
for name, fn in (
    ("MATH (current)", lambda a, b, c: orig_sdpa(a, b, c, attn_mask=allm, enable_gqa=True)),
    ("exact chunked (new)", lambda a, b, c: exact_attention(a, b, c, attn_mask=allm)),
):
    go = torch.randn(B, Hq, L, D, device=dev, dtype=dt)
    for it in range(3):
        a, b2, c2 = (t.detach().clone().requires_grad_() for t in (q, k, v))
        torch.xpu.synchronize()
        torch.xpu.reset_peak_memory_stats()
        base = torch.xpu.memory_allocated()
        t = time.time()
        if name.startswith("MATH"):
            with sdpa_kernel([SDPBackend.MATH]):
                o = fn(a, b2, c2)
        else:
            o = fn(a, b2, c2)
        o.backward(go)
        torch.xpu.synchronize()
        dt_ms = (time.time() - t) * 1000
    peak = (torch.xpu.max_memory_allocated() - base) / 2 ** 30
    print(f"{name:22s} fwd+bwd {dt_ms:7.1f} ms   extra peak VRAM {peak:5.2f} GB", flush=True)

print("RESULT:", "PASS" if ok else "FAIL")
