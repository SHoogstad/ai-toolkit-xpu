"""CUDA -> Intel XPU compatibility shim for ai-toolkit.

Loaded at interpreter start via a .pth file when AITK_XPU=1. ai-toolkit (and
the configs its UI writes) assume device "cuda"; this remaps device strings,
the torch.cuda namespace and autocast onto torch.xpu so the trainer runs on an
Intel Arc GPU without editing every call site.
"""
import functools
import os
import sys
import weakref


def _install():
    import torch

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("[xpu_shim] torch.xpu not available, shim disabled", file=sys.stderr)
        return

    xpu = torch.xpu

    # libraries that pick their backend at import time from
    # torch.cuda.is_available() must be imported before we alias it
    try:
        import bitsandbytes  # noqa: F401
    except Exception as e:
        print(f"[xpu_shim] bitsandbytes preload failed: {e!r}", file=sys.stderr)

    def fix(dev):
        if isinstance(dev, str):
            if dev.startswith("cuda"):
                return "xpu" + dev[4:]
            return dev
        if isinstance(dev, _OrigDevice) and dev.type == "cuda":
            return _OrigDevice("xpu", dev.index) if dev.index is not None else _OrigDevice("xpu")
        if isinstance(dev, int) and not isinstance(dev, bool):
            # bare ints mean "cuda:<n>" in torch
            return _OrigDevice("xpu", dev)
        return dev

    # ---- torch.device ---------------------------------------------------
    _OrigDevice = torch.device

    class _DeviceMeta(type):
        def __instancecheck__(cls, obj):
            return isinstance(obj, _OrigDevice)

        def __subclasscheck__(cls, sub):
            return issubclass(sub, _OrigDevice)

        def __call__(cls, *args, **kwargs):
            if args:
                args = (fix(args[0]),) + args[1:]
            if "type" in kwargs:
                kwargs["type"] = fix(kwargs["type"])
            return _OrigDevice(*args, **kwargs)

    class device(metaclass=_DeviceMeta):
        pass

    device.__module__ = "torch"
    device.__qualname__ = "device"
    torch.device = device

    # ---- functions taking a device kwarg --------------------------------
    import torch.jit._builtins as _jit_builtins
    import torch.utils._device as _torch_device_ctx

    # `with torch.device("meta"):` only redirects factory calls whose function
    # object is in this cached set. Build it from the originals before we wrap
    # them, or meta-device model init silently runs on CPU (transformers then
    # leaves rotary inv_freq buffers uninitialized -> NaN text encoders).
    _torch_device_ctx._device_constructors()

    def wrap_device_kw(fn, pos=None):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = fix(kwargs["device"])
            if pos is not None and len(args) > pos:
                args = args[:pos] + (fix(args[pos]),) + args[pos + 1:]
            return fn(*args, **kwargs)
        return inner

    for name in (
        "zeros", "ones", "empty", "full", "rand", "randn", "randint", "randperm",
        "arange", "linspace", "logspace", "tensor", "as_tensor", "eye",
        "zeros_like", "ones_like", "empty_like", "full_like", "rand_like",
        "randn_like", "randint_like", "empty_strided", "tril_indices",
        "triu_indices", "normal", "scalar_tensor", "asarray", "frombuffer",
    ):
        if hasattr(torch, name):
            orig = getattr(torch, name)
            wrapped = wrap_device_kw(orig)
            # let TorchScript keep resolving the wrapper to the aten op
            op = _jit_builtins._find_builtin(orig)
            if op is not None:
                _jit_builtins._register_builtin(wrapped, op)
            setattr(torch, name, wrapped)

    _OrigGenerator = torch.Generator

    class _GenMeta(type):
        def __instancecheck__(cls, obj):
            return isinstance(obj, _OrigGenerator)

        def __call__(cls, device="cpu", *a, **k):
            return _OrigGenerator(fix(device), *a, **k)

    class Generator(metaclass=_GenMeta):
        pass

    torch.Generator = Generator

    # Tensor / Module movement
    def fix_to_args(args, kwargs):
        if args and isinstance(args[0], (str, _OrigDevice, int)) and not isinstance(args[0], bool):
            args = (fix(args[0]),) + args[1:]
        if "device" in kwargs:
            kwargs["device"] = fix(kwargs["device"])
        return args, kwargs

    _t_to = torch.Tensor.to

    def t_to(self, *args, **kwargs):
        args, kwargs = fix_to_args(args, kwargs)
        return _t_to(self, *args, **kwargs)

    torch.Tensor.to = t_to

    def t_cuda(self, device=None, non_blocking=False, **kw):
        return _t_to(self, fix(device) if device is not None else "xpu", non_blocking=non_blocking)

    torch.Tensor.cuda = t_cuda
    torch.Tensor.is_cuda = property(lambda self: self.device.type in ("cuda", "xpu"))

    for name in ("new_zeros", "new_ones", "new_empty", "new_full", "new_tensor"):
        setattr(torch.Tensor, name, wrap_device_kw(getattr(torch.Tensor, name)))

    _m_to = torch.nn.Module.to

    def m_to(self, *args, **kwargs):
        args, kwargs = fix_to_args(args, kwargs)
        return _m_to(self, *args, **kwargs)

    torch.nn.Module.to = m_to
    torch.nn.Module.cuda = lambda self, device=None: _m_to(self, fix(device) if device is not None else "xpu")

    _load = torch.load

    @functools.wraps(_load)
    def load(*args, **kwargs):
        if "map_location" in kwargs:
            kwargs["map_location"] = fix(kwargs["map_location"])
        return _load(*args, **kwargs)

    torch.load = load

    # ---- autocast -------------------------------------------------------
    _OrigAutocast = torch.amp.autocast

    class autocast(_OrigAutocast):
        def __init__(self, device_type="cuda", *args, **kwargs):
            super().__init__(fix(device_type), *args, **kwargs)

    torch.amp.autocast = autocast
    torch.autocast = autocast
    torch.amp.autocast_mode.autocast = autocast

    # ---- torch.cuda namespace -> torch.xpu ------------------------------
    cuda = torch.cuda
    for name in (
        "is_available", "device_count", "current_device", "set_device", "synchronize",
        "empty_cache", "memory_allocated", "max_memory_allocated", "memory_reserved",
        "max_memory_reserved", "reset_peak_memory_stats", "memory_stats", "mem_get_info",
        "get_device_name", "get_device_properties", "get_device_capability",
        "manual_seed", "manual_seed_all", "seed", "seed_all", "initial_seed",
        "get_rng_state", "set_rng_state", "get_rng_state_all", "set_rng_state_all",
        "current_stream", "default_stream", "stream", "Stream", "Event", "device",
        "is_initialized", "init", "is_bf16_supported", "StreamContext",
    ):
        if hasattr(xpu, name):
            target = getattr(xpu, name)
            if callable(target) and not isinstance(target, type):
                # distinct function object: torch._dynamo keys handlers by
                # identity and asserts if cuda.X and xpu.X are the same object
                target = functools.wraps(target)(
                    lambda *a, _f=target, **k: _f(*a, **k))
            setattr(cuda, name, target)
    cuda.is_bf16_supported = lambda *a, **k: True
    cuda.ipc_collect = lambda: None
    cuda.reset_max_memory_allocated = xpu.reset_peak_memory_stats
    cuda.get_arch_list = lambda: []
    if not hasattr(xpu, "mem_get_info"):
        def mem_get_info(dev=None):
            total = xpu.get_device_properties(dev or 0).total_memory
            return total - xpu.memory_reserved(dev), total
        cuda.mem_get_info = mem_get_info
    # device-capability gates (fp8, flash paths) -> report Ampere-class so code
    # takes the generic bf16 paths rather than Hopper/Blackwell kernels
    cuda.get_device_capability = lambda *a, **k: (8, 0)

    class _Props:
        major, minor = 8, 0

        def __init__(self, p):
            self._p = p
            self.multi_processor_count = getattr(p, "gpu_subslice_count", 20)

        def __getattr__(self, name):
            return getattr(self._p, name)

    _xpu_props = xpu.get_device_properties
    cuda.get_device_properties = lambda *a, **k: _Props(_xpu_props(*a, **k))

    import torch.cuda.amp as cuda_amp

    class cuda_autocast(_OrigAutocast):
        def __init__(self, enabled=True, dtype=torch.float16, cache_enabled=True):
            super().__init__("xpu", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled)

    cuda_amp.autocast = cuda_autocast
    cuda.amp.autocast = cuda_autocast

    # ---- pinned host memory budget --------------------------------------
    # The xe/Level Zero stack keeps only part of pinned host memory mapped for
    # the GPU, and how much varies with system state (~8 GB when idle, under
    # 4 GB on a busy host). Past that, every H2D copy from a pinned buffer
    # remaps it: layer offloading drops from 14 GB/s to ~0.6 GB/s, and the remap
    # VM jobs can trip the driver's job timeout (DEVICE_LOST). Pageable copies go
    # through driver staging at ~4-8 GB/s at any size. Mixing the two is worst of
    # all: pinned buffers plus the driver's own staging buffers overflow the
    # mapping, and Krea 2's real offloaded weights (2 GB pinned + 7.5 GB
    # pageable) copied at 0.4 GB/s vs 4.5 GB/s all-pageable. So don't pin by
    # default; AITK_XPU_PIN_BUDGET_GB allows a budget for experiments.
    budget = int(float(os.environ.get("AITK_XPU_PIN_BUDGET_GB", "0")) * 1024 ** 3)
    pinned = {"bytes": 0}
    _pin_memory = torch.Tensor.pin_memory

    def _unpinned(nbytes):
        pinned["bytes"] -= nbytes

    def pin_memory(self, *args, **kwargs):
        nbytes = self.untyped_storage().nbytes()
        if pinned["bytes"] + nbytes > budget:
            return self
        out = _pin_memory(self, *args, **kwargs)
        pinned["bytes"] += nbytes
        weakref.finalize(out, _unpinned, nbytes)
        return out

    torch.Tensor.pin_memory = pin_memory

    # ---- VRAM cap -------------------------------------------------------
    # The xe driver never refuses a device allocation: past physical VRAM it
    # silently evicts to system memory (~20x slower), so torch's caching
    # allocator keeps growing instead of reusing its cache. Capping it below
    # physical VRAM makes it free cached blocks first and raise a real OOM
    # when the model doesn't fit. Leave ~10% for Level Zero / kernels.
    # AITK_XPU_MEM_FRACTION=1 restores the old (overcommitting) behaviour.
    frac = float(os.environ.get("AITK_XPU_MEM_FRACTION", "0.9"))
    if 0 < frac < 1:
        xpu.set_per_process_memory_fraction(frac)

    # ---- masked attention -------------------------------------------------
    # On XPU any attn_mask sends SDPA to the MATH backend, which materializes
    # the full fp32 score matrix (and its gradient): GBs per call at DiT
    # sequence lengths (Krea 2: 1.1 GB per score tensor), which is what forces
    # heavy layer offloading. Route masked calls whose score matrix exceeds
    # AITK_XPU_ATTN_MB (default 512) through xpu_attention.exact_attention:
    # chunked, recomputed in backward, MATH-level fp32 numerics (verified
    # against float64 on Krea 2's extreme block-0 logits, where the fused
    # kernel's gradients are off by 6x). Smaller masked calls stay on MATH,
    # unmasked calls stay on the fused kernel. AITK_XPU_ATTN_MB=0 disables.
    # Within those, xpu_attention.hybrid_attention still uses the fused kernel
    # when it's safe: a tail-padding mask is sliced away (exactly equivalent)
    # and logits are moderate (max|q|*max|k|*sqrt(D) <= AITK_XPU_FLASH_MAX,
    # default 2500, where the fused kernel's gradients stay at bf16-level
    # error); only extreme calls (Krea 2 block 0, ~4e6) pay for the exact one.
    attn_mb = float(os.environ.get("AITK_XPU_ATTN_MB", "512"))
    flash_max = float(os.environ.get("AITK_XPU_FLASH_MAX", "2500"))
    if attn_mb > 0:
        import torch.nn.functional as F
        import xpu_attention

        _sdpa = F.scaled_dot_product_attention

        def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                         is_causal=False, scale=None, enable_gqa=False, **kw):
            if (not kw and xpu_attention.supported(query, key, value, attn_mask, dropout_p, is_causal)
                    and (enable_gqa or query.shape[1] == key.shape[1])
                    and query.shape[0] * query.shape[1] * query.shape[2] * key.shape[2] * 4
                    > attn_mb * 1024 ** 2):
                return xpu_attention.hybrid_attention(query, key, value, attn_mask, scale, enable_gqa,
                                                      _sdpa, flash_max)
            return _sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                         is_causal=is_causal, scale=scale, enable_gqa=enable_gqa, **kw)

        F._xpu_orig_sdpa = _sdpa
        F.scaled_dot_product_attention = scaled_dot_product_attention

    # ---- layer-offload prefetch -------------------------------------------
    # ai-toolkit's offload manager is imported long after this shim runs, so
    # patch it on import: xpu_prefetch copies the next AITK_XPU_PREFETCH
    # predicted layers (default 3, 0 disables) on a background thread while
    # the current one computes (see xpu_prefetch.py).
    pf_depth = int(os.environ.get("AITK_XPU_PREFETCH", "3"))
    if pf_depth > 0:
        import importlib.abc
        import importlib.util

        target_mod = "toolkit.memory_management.manager_modules"

        class _PrefetchHook(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name != target_mod:
                    return None
                sys.meta_path.remove(self)
                try:
                    spec = importlib.util.find_spec(name)
                except Exception:
                    return None
                if spec is None or spec.loader is None:
                    return spec
                exec_module = spec.loader.exec_module

                def exec_and_patch(module):
                    exec_module(module)
                    try:
                        import xpu_prefetch

                        xpu_prefetch.install(module, pf_depth)
                    except Exception as e:  # never break the trainer over an optimization
                        print(f"[xpu_shim] layer prefetch not installed: {e!r}", file=sys.stderr)

                spec.loader.exec_module = exec_and_patch
                return spec

        sys.meta_path.insert(0, _PrefetchHook())

    # ---- non-finite gradient guard ----------------------------------------
    # One inf/NaN gradient is enough to destroy a whole LoRA: global-norm
    # clipping turns the total norm into NaN and multiplies it into *every*
    # gradient, the optimizer writes NaN into all weights, and every sample
    # after that decodes to black while the logged loss stays finite. XPU
    # kernels hit this where CUDA doesn't (e.g. fused SDPA backward on Krea 2's
    # huge block-0 logits), so before every optimizer step: if any gradient is
    # non-finite, drop all gradients for that step (the optimizer then skips
    # the update, momentum included) and warn; abort after too many in a row.
    # AITK_XPU_NAN_GUARD=0 disables it.
    if os.environ.get("AITK_XPU_NAN_GUARD", "1") == "1":
        from torch.optim.optimizer import register_optimizer_step_pre_hook

        max_bad = int(os.environ.get("AITK_XPU_NAN_GUARD_MAX", "10"))
        guard = {"bad": 0, "total": 0}

        def nan_guard(optimizer, args, kwargs):
            grads = [p.grad for group in optimizer.param_groups for p in group["params"]
                     if p.grad is not None]
            if not grads:
                return None
            ok = torch.stack([torch.isfinite(g).all() for g in grads]).all().item()
            if ok:
                guard["bad"] = 0
                return None
            guard["bad"] += 1
            guard["total"] += 1
            print(f"[xpu_shim] non-finite gradients, skipping this optimizer step "
                  f"({guard['bad']} in a row, {guard['total']} total)", file=sys.stderr, flush=True)
            if guard["bad"] >= max_bad:
                raise RuntimeError(
                    f"non-finite gradients for {guard['bad']} optimizer steps in a row; stopping so the "
                    "weights aren't corrupted (AITK_XPU_NAN_GUARD_MAX raises the limit)")
            for group in optimizer.param_groups:
                for p in group["params"]:
                    p.grad = None
            return None

        register_optimizer_step_pre_hook(nan_guard)

    # ---- safetensors ----------------------------------------------------
    try:
        import safetensors.torch as st

        st.load_file = wrap_device_kw(st.load_file, pos=1)
        import safetensors

        _safe_open = safetensors.safe_open

        def safe_open(filename, framework, device="cpu", **kwargs):
            return _safe_open(filename, framework, device=fix(device), **kwargs)

        safetensors.safe_open = safe_open
        st.safe_open = safe_open
    except Exception as e:  # pragma: no cover
        print(f"[xpu_shim] safetensors patch skipped: {e}", file=sys.stderr)

    print(f"[xpu_shim] cuda -> xpu ({xpu.get_device_name(0)}), vram cap {frac:g}, "
          f"pin budget {budget / 1024 ** 3:g} GB", file=sys.stderr)


if os.environ.get("AITK_XPU") == "1":
    try:
        _install()
    except Exception as e:  # never break the interpreter
        print(f"[xpu_shim] failed: {e!r}", file=sys.stderr)
