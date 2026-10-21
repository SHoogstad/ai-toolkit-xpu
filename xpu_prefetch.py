"""Background weight prefetch for ai-toolkit's layer offloading on XPU.

ai-toolkit's OstrisLinear offload path copies a layer's quantized buffers to the
GPU on the training thread when the layer is called, then computes. On XPU those
copies come from pageable memory (pinned memory is disabled, see xpu_shim), so
each copy blocks the thread: copy and compute alternate instead of overlapping.

This learns the order layers are called in (an order-2 "after A, B comes C"
table, which captures both the forward pass and the backward pass's
recomputation) and has a background thread copy the next few predicted layers
on a separate stream while the GPU computes the current one. PyTorch releases
the GIL during copies, so they really run in parallel. A layer whose copy is
ready waits only on its own event. Anything unexpected -- a misprediction,
changed source buffers (requantize during merge), CPU activations, an unusual
call signature -- falls back to ai-toolkit's original path, so correctness never
depends on a prediction.
"""
import collections
import queue
import sys
import threading

import torch

# calls of context used to predict the next layer; must exceed the number of
# offloaded linears in one transformer block (Krea 2: ~10 at 60% offload)
HISTORY = 16


class _Slot:
    __slots__ = ("done", "src", "bufs", "bias", "event")

    def __init__(self, src):
        self.done = threading.Event()
        self.src = src  # identities of the cpu tensors this copy was made from
        self.bufs = None
        self.bias = None
        self.event = None


class Prefetcher:
    def __init__(self, device, depth):
        self.device = device
        self.depth = depth
        self.stream = torch.xpu.Stream(device=device)
        self.table = {}  # tuple(last HISTORY managers) -> next layer manager
        self.hist = collections.deque(maxlen=HISTORY)
        self.slots = collections.OrderedDict()  # mgr -> _Slot, oldest first
        self.q = queue.Queue()
        self.stats = {"hit": 0, "miss": 0, "stale": 0, "dropped": 0}
        threading.Thread(target=self._worker, name="xpu-prefetch", daemon=True).start()

    # ---- background thread ------------------------------------------------
    def _worker(self):
        while True:
            mgr, slot, bufs_cpu, bias_cpu = self.q.get()
            try:
                with torch.xpu.stream(self.stream):
                    slot.bufs = {n: t.to(self.device, non_blocking=True) for n, t in bufs_cpu.items()}
                    slot.bias = bias_cpu.to(self.device, non_blocking=True) if bias_cpu is not None else None
                    slot.event = torch.xpu.Event()
                    slot.event.record(self.stream)
            except Exception as e:  # never break training: the layer will fall back
                print(f"[xpu_prefetch] copy failed, falling back: {e!r}", file=sys.stderr)
                slot.bufs = None
            finally:
                slot.done.set()

    # ---- main thread ------------------------------------------------------
    def _schedule(self, mgr):
        if mgr in self.slots:
            return
        src = _cpu_sources(mgr)
        if src is None:
            return
        bufs_cpu, bias_cpu = src
        slot = _Slot(_identity(bufs_cpu, bias_cpu))
        self.slots[mgr] = slot
        self.q.put((mgr, slot, bufs_cpu, bias_cpu))
        # bound VRAM held by mispredicted copies
        while len(self.slots) > 2 * self.depth + 2:
            old, s = self.slots.popitem(last=False)
            s.done.wait()
            self.stats["dropped"] += 1

    def on_call(self, mgr):
        """Record the transition into mgr and queue copies of the next layers.

        The context is the last HISTORY calls, not just the last one or two:
        with gradient checkpointing, the forward pass and the backward's
        recomputation run the same layers within a block but diverge at its
        end (next block vs previous block). A short context can't tell them
        apart (about a third of the copies were mispredicted and wasted, even
        keyed on grad mode, since non-reentrant checkpointing runs both with
        grad enabled); a history longer than one block's offloaded layers
        always reaches into the preceding block, which differs."""
        if self.hist:
            self.table[tuple(self.hist)] = mgr
        self.hist.append(mgr)
        ctx = collections.deque(self.hist, maxlen=self.hist.maxlen)
        for _ in range(self.depth):
            nxt = self.table.get(tuple(ctx))
            if nxt is None:
                break
            self._schedule(nxt)
            ctx.append(nxt)

    def take(self, mgr):
        slot = self.slots.pop(mgr, None)
        if slot is None:
            self.stats["miss"] += 1
            return None
        slot.done.wait()
        src = _cpu_sources(mgr)
        if slot.bufs is None or src is None or _identity(*src) != slot.src:
            self.stats["stale"] += 1
            return None
        self.stats["hit"] += 1
        return slot


def _cpu_sources(mgr):
    module = mgr.module
    bufs = {n: b for n, b in module._buffers.items() if b is not None and b.device.type == "cpu"}
    bias = module._parameters.get("bias", None)
    bias_cpu = bias.data if bias is not None and bias.data.device.type == "cpu" else None
    if not bufs and bias_cpu is None:
        return None  # resident on the device, nothing to prefetch
    return bufs, bias_cpu


def _identity(bufs, bias):
    return tuple(sorted((n, id(t), t.data_ptr()) for n, t in bufs.items())) + (
        (id(bias), bias.data_ptr()) if bias is not None else (None,),)


_prefetchers = {}


def _prefetcher(device, depth):
    p = _prefetchers.get(device)
    if p is None:
        p = _prefetchers[device] = Prefetcher(device, depth)
    return p


def wrap_manager(mgr, depth):
    """Wrap an OstrisLinearLayerMemoryManager's installed forward."""
    module = mgr.module
    if hasattr(module, "ara_lora_ref"):
        target, attr = module.ara_lora_ref(), "org_forward"
    else:
        target, attr = module, "forward"
    orig = getattr(target, attr)

    def forward(x, *args, **kwargs):
        device = mgr.manager.process_device
        # compare by type: torch.device("xpu") != torch.device("xpu:0")
        if args or kwargs or device.type != "xpu" or x.device.type != "xpu":
            return orig(x, *args, **kwargs)
        device = x.device
        pf = _prefetcher(device, depth)
        pf.on_call(mgr)
        slot = pf.take(mgr)
        if slot is None:
            return orig(x)
        cs = torch.xpu.current_stream(device)
        cs.wait_event(slot.event)
        # allocated on the prefetch stream, used on the compute stream (and
        # possibly saved for backward): keep the allocator from recycling them
        for t in slot.bufs.values():
            t.record_stream(cs)
        if slot.bias is not None:
            slot.bias.record_stream(cs)
        cpu_bufs = {n: module._buffers[n] for n in slot.bufs}
        bias = module._parameters.get("bias", None)
        bias_cpu = bias.data if slot.bias is not None else None
        for n, t in slot.bufs.items():
            module._buffers[n] = t
        if slot.bias is not None:
            bias.data = slot.bias
        try:
            return mgr._original_forward(x)
        finally:
            for n, t in cpu_bufs.items():
                module._buffers[n] = t
            if bias_cpu is not None:
                bias.data = bias_cpu

    setattr(target, attr, forward)


def install(manager_modules, depth):
    """Patch OstrisLinearLayerMemoryManager so every new instance is wrapped."""
    cls = manager_modules.OstrisLinearLayerMemoryManager
    orig_init = cls.__init__

    def __init__(self, module, manager):
        orig_init(self, module, manager)
        wrap_manager(self, depth)

    cls.__init__ = __init__

    import atexit

    def report():
        for d, p in _prefetchers.items():
            s = p.stats
            total = s["hit"] + s["miss"] + s["stale"]
            if total:
                print(f"[xpu_prefetch] {d}: {s['hit']}/{total} layer copies prefetched "
                      f"({100 * s['hit'] / total:.0f}%), {s['stale']} stale, {s['dropped']} mispredicted",
                      file=sys.stderr, flush=True)

    atexit.register(report)


def stats():
    return {str(d): dict(p.stats) for d, p in _prefetchers.items()}
