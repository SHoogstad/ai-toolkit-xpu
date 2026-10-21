"""xpu_prefetch with fake offloaded layers called in a checkpointing pattern.

Each "block" runs its layers in order during the forward pass, and again in
order during the backward's recomputation, but blocks are recomputed in
reverse. That is the pattern that defeats a short-context predictor.
"""
import sys

import torch

import xpu_prefetch

failures = []
DEV = torch.device("xpu")


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        failures.append(name)


class FakeQuantLinear(torch.nn.Module):
    """Weights as a cpu buffer, like an offloaded OstrisLinear."""

    def __init__(self, n):
        super().__init__()
        self.register_buffer("qdata", torch.randn(n, n, dtype=torch.bfloat16))

    def forward(self, x):
        return x @ self.qdata


class FakeManager:
    """The bits of OstrisLinearLayerMemoryManager that xpu_prefetch uses."""

    def __init__(self, module):
        self.module = module
        self.manager = type("M", (), {"process_device": DEV})()
        self._original_forward = module.forward
        self.fallbacks = 0

        def mm_forward(x):  # the original offload path: copy, then compute
            self.fallbacks += 1
            cpu = self.module._buffers["qdata"]
            self.module._buffers["qdata"] = cpu.to(DEV)
            try:
                return self._original_forward(x)
            finally:
                self.module._buffers["qdata"] = cpu

        module.forward = mm_forward


N_BLOCKS, PER_BLOCK, DIM = 6, 4, 256
blocks = [[FakeManager(FakeQuantLinear(DIM)) for _ in range(PER_BLOCK)] for _ in range(N_BLOCKS)]
for blk in blocks:
    for mgr in blk:
        xpu_prefetch.wrap_manager(mgr, depth=3)


def run_block(blk, x):
    for mgr in blk:
        x = torch.tanh(mgr.module(x)) * 0.5
    return x


def micro_step(x):
    outs = []
    for blk in blocks:  # forward
        x = run_block(blk, x)
    for blk in reversed(blocks):  # backward recomputation, blocks in reverse
        outs.append(run_block(blk, x))
    return x, outs


def reference(x):
    """Same math with weights moved directly, no offloading or prefetch."""
    def blk_ref(blk, x):
        for mgr in blk:
            x = torch.tanh(x @ mgr.module._buffers["qdata"].to(DEV)) * 0.5
        return x
    outs = []
    for blk in blocks:
        x = blk_ref(blk, x)
    for blk in reversed(blocks):
        outs.append(blk_ref(blk, x))
    return x, outs


torch.manual_seed(0)
x0 = torch.randn(32, DIM, device=DEV, dtype=torch.bfloat16)
ref_x, ref_outs = reference(x0)
pf_stats = []
for it in range(4):
    out_x, outs = micro_step(x0)
    torch.xpu.synchronize()
    same = torch.equal(out_x, ref_x) and all(torch.equal(a, b) for a, b in zip(outs, ref_outs))
    check(f"micro-step {it}: outputs identical to direct computation", same)
    pf_stats.append(dict(next(iter(xpu_prefetch._prefetchers.values())).stats))

s0, s3 = pf_stats[0], pf_stats[3]
calls_per_step = 2 * N_BLOCKS * PER_BLOCK
new_misses = s3["miss"] - pf_stats[1]["miss"]
new_hits = s3["hit"] - pf_stats[1]["hit"]
new_dropped = s3["dropped"] - pf_stats[1]["dropped"]
check("after the first micro-step every layer is prefetched", new_misses == 0 and new_hits == 2 * calls_per_step,
      f"hits {new_hits}/{2 * calls_per_step}, misses {new_misses} in steps 2-3")
check("no wasted (mispredicted) copies after learning", new_dropped == 0, f"dropped {new_dropped}")

# changed source buffers (e.g. requantize) must not reuse a stale prefetch
blk0 = blocks[0][0]
blk0.module._buffers["qdata"] = torch.randn(DIM, DIM, dtype=torch.bfloat16)
ref_x, ref_outs = reference(x0)
out_x, outs = micro_step(x0)
torch.xpu.synchronize()
check("replaced weights are picked up, not a stale prefetch",
      torch.equal(out_x, ref_x) and all(torch.equal(a, b) for a, b in zip(outs, ref_outs)))

print(f"\n{len(failures)} failure(s)" if failures else "\nall prefetch tests passed")
sys.exit(1 if failures else 0)
