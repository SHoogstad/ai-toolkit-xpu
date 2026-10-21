import sys, torch, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
sys.path.insert(0, "/tmp")
import xpu_attention as xa
orig = F.scaled_dot_product_attention
torch.manual_seed(0)
B, Hq, Hkv, L, D = 1, 48, 12, 2560, 128
n = 2384
calls = {"fused": 0, "exact": 0}
_ex = xa.exact_attention
def ex(*a, **k):
    calls["exact"] += 1
    return _ex(*a, **k)
xa.exact_attention = ex
def fl(*a, **k):
    calls["fused"] += 1
    return orig(*a, **k)
def check(tag, amp, mask):
    q, k, v = (torch.randn(B, h, L, D, device="xpu", dtype=torch.bfloat16) * s for h, s in ((Hq, amp), (Hkv, amp), (Hkv, 1)))
    go = torch.randn(B, Hq, L, D, device="xpu", dtype=torch.bfloat16)
    def g(fn):
        a, b, c = (t.detach().clone().requires_grad_() for t in (q, k, v))
        o = fn(a, b, c); o.backward(go)
        return [t.float() for t in (o, a.grad, b.grad, c.grad)]
    def ref(a, b, c):
        with sdpa_kernel([SDPBackend.MATH]):
            return orig(a, b, c, attn_mask=mask, enable_gqa=True)
    calls.update(fused=0, exact=0)
    mine = g(lambda a, b, c: xa.hybrid_attention(a, b, c, mask, None, True, fl, 2500.0))
    theirs = g(ref)
    rows = mask.any(-1)[0, 0] if mask is not None else torch.ones(L, dtype=torch.bool, device="xpu")
    errs = []
    for nm, a, b in zip(("out", "dq", "dk", "dv"), mine, theirs):
        if nm in ("out", "dq"):  # padded query rows: MATH gives 0 too, compare valid rows
            a, b = a[:, :, rows], b[:, :, rows]
        errs.append(f"{nm} {((a - b).norm() / b.norm()).item():.2e}")
    padzero = bool((mine[0][:, :, ~rows] == 0).all()) if (~rows).any() else True
    print(f"{tag:42s} path {calls}  finite {all(bool(torch.isfinite(t).all()) for t in mine)}  padded rows zero {padzero}  vs MATH: {', '.join(errs)}", flush=True)
pad = torch.zeros(B, L, dtype=torch.bool, device="xpu"); pad[:, :n] = True
tail = pad[:, None, :, None] & pad[:, None, None, :]
check("tail padding, moderate (expect fused)", 1.0, tail)
check("tail padding, extreme x30 (expect exact)", 30.0, tail)
odd = tail.clone(); odd[..., 5, 7] = False
check("non-tail mask (expect exact)", 1.0, odd)
check("all-true mask, moderate (expect fused)", 1.0, torch.ones(B, 1, L, L, dtype=torch.bool, device="xpu"))
