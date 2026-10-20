"""Let Krea 2 use unmasked SDPA on XPU when nothing is actually masked.

Krea 2 always passes an attention mask: text padding plus a pad of the joint
sequence to a multiple of 256 (only there to stabilize torch.compile shapes).
On XPU any attn_mask forces the MATH SDPA backend, which materializes the full
(heads x L x L) attention matrix: ~6x slower than the fused kernel and several
GB of extra VRAM during training. With AITK_XPU=1 this skips the alignment pad
and drops masks that are all True (e.g. batch 1), which is numerically the same
attention. CUDA behaviour is unchanged.

Idempotent; fails soft (warns) if upstream code moved.
usage: python3 krea2_xpu_attention.py /app/ai-toolkit
"""
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "/app/ai-toolkit"
path = os.path.join(root, "extensions_built_in/diffusion_models/krea2/src/mmdit.py")
MARK = "# AITK_XPU_ATTN_PATCH"

EDITS = [
    (
        "import math\n",
        "import math\nimport os  " + MARK + "\n",
    ),
    (
        "        txtmask = _mask(mask[:, : context.shape[1]])\n",
        "        txtmask = _mask(mask[:, : context.shape[1]])\n"
        "        if os.environ.get(\"AITK_XPU\") == \"1\" and bool(mask[:, : context.shape[1]].all()):\n"
        "            txtmask = None  " + MARK + ": unmasked SDPA on XPU\n",
    ),
    (
        "        _padlen = (-fulllen) % 256\n",
        "        _padlen = (-fulllen) % 256\n"
        "        if os.environ.get(\"AITK_XPU\") == \"1\":\n"
        "            _padlen = 0  " + MARK + ": the pad only exists for torch.compile\n",
    ),
    (
        "        padmask = mask  # (B, L) key-padding mask, incl. the 256-alignment pad\n"
        "        mask = _mask(mask)\n",
        "        padmask = mask  # (B, L) key-padding mask, incl. the 256-alignment pad\n"
        "        mask = _mask(mask)\n"
        "        if (\n"
        "            os.environ.get(\"AITK_XPU\") == \"1\"\n"
        "            and reflen == 0\n"
        "            and ref_kv_cache is None\n"
        "            and bool(padmask.all())\n"
        "        ):\n"
        "            mask = None  " + MARK + ": unmasked SDPA on XPU\n",
    ),
]

src = open(path).read()
if MARK in src:
    print(f"[krea2_xpu_attention] already patched: {path}")
    sys.exit(0)
missing = [old for old, _ in EDITS if src.count(old) != 1]
if missing:
    print(f"[krea2_xpu_attention] WARNING: upstream changed, not patching {path}; "
          f"unmatched: {[m.strip()[:60] for m in missing]}", file=sys.stderr)
    sys.exit(0)
for old, new in EDITS:
    src = src.replace(old, new)
open(path, "w").write(src)
print(f"[krea2_xpu_attention] patched {path}")
