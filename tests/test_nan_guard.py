"""xpu_shim non-finite gradient guard (needs the shim active: AITK_XPU=1)."""
import sys

import torch

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        failures.append(name)


m = torch.nn.Linear(8, 8).to("cuda")  # "cuda" is mapped to xpu by the shim
opt = torch.optim.AdamW(m.parameters(), lr=1e-2)


def step(inject):
    opt.zero_grad()
    m(torch.randn(4, 8, device="cuda")).sum().backward()
    if inject:
        m.weight.grad[0, 0] = float("inf")
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    before = [p.detach().clone() for p in m.parameters()]
    opt.step()
    finite = all(bool(torch.isfinite(p).all()) for p in m.parameters())
    changed = max((p - b).abs().max().item() for p, b in zip(m.parameters(), before))
    return finite, changed


finite, changed = step(False)
check("normal step updates weights", finite and changed > 0, f"max change {changed:.3g}")
finite, changed = step(True)
check("step with an inf gradient is skipped", finite and changed == 0, f"finite {finite}, max change {changed:.3g}")
finite, changed = step(False)
check("training continues after a skipped step", finite and changed > 0, f"max change {changed:.3g}")
try:
    for _ in range(12):
        step(True)
    check("aborts after too many bad steps in a row", False, "no exception raised")
except RuntimeError as e:
    check("aborts after too many bad steps in a row", "non-finite gradients" in str(e), str(e)[:60])

print(f"\n{len(failures)} failure(s)" if failures else "\nall guard tests passed")
sys.exit(1 if failures else 0)
