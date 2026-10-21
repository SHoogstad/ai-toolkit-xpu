"""End-to-end check that LoRA training on a model works on this GPU.

Runs a job config for a few steps, then checks:
  1. how many optimizer steps the shim's non-finite-gradient guard skipped
  2. the saved LoRA is finite (and not all zeros)
  3. a sample generated *with* the trained LoRA is a real image (not black)

usage (inside the container):
  python3 validate_model.py <job_config.json> [steps=30]
env: OVERRIDE='{"model": {...}, "train": {...}}' merged into the process config
Exit code 0 = pass, 1 = fail.
"""
import glob
import json
import os
import subprocess
import sys

src = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
name = "validate_" + os.path.splitext(os.path.basename(os.path.dirname(os.path.abspath(src))))[0]
out_dir = f"/app/ai-toolkit/output/{name}"
subprocess.run(["rm", "-rf", out_dir], check=True)

cfg = json.load(open(src))
p = cfg["config"]["process"][0]
cfg["config"]["name"] = name
p.pop("sqlite_db_path", None)
p["logging"]["use_ui_logger"] = False
p["train"]["steps"] = steps
p["train"]["skip_first_sample"] = True
p["train"]["disable_sampling"] = False
p["save"]["save_every"] = steps
p["datasets"] = [d for d in p["datasets"] if not d.get("is_reg")]
s = p["sample"]
s["samples"] = s["samples"][:1]
s["width"] = s["height"] = 512
s["sample_steps"] = 12
s["sample_every"] = steps
for sect, vals in json.loads(os.environ.get("OVERRIDE", "{}")).items():
    p[sect].update(vals)
os.makedirs("/tmp/probe", exist_ok=True)
cfg_path = f"/tmp/probe/{name}.json"
json.dump(cfg, open(cfg_path, "w"))

print(f"[validate] {p['model'].get('arch')} {p['model']['name_or_path']}: {steps} steps", flush=True)
proc = subprocess.run([sys.executable, "run.py", cfg_path], cwd="/app/ai-toolkit",
                      capture_output=True, text=True)
log = proc.stdout + proc.stderr
open(f"/tmp/probe/{name}.log", "w").write(log)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

fails = []
skipped = log.count("non-finite gradients, skipping")
print(f"[validate] run exit code {proc.returncode}; guard skipped {skipped} optimizer step(s)")
if proc.returncode != 0:
    fails.append("training run failed (see /tmp/probe/%s.log)" % name)
    print("\n".join(log.splitlines()[-15:]))
if skipped:
    fails.append(f"{skipped} optimizer steps had non-finite gradients")

loras = sorted(glob.glob(f"{out_dir}/*.safetensors"))
if not loras:
    fails.append("no LoRA saved")
else:
    sd = load_file(loras[-1])
    bad = [k for k, v in sd.items() if not torch.isfinite(v.float()).all()]
    up = [v for k, v in sd.items() if "lora_B" in k or "lora_up" in k]
    moved = max((v.float().abs().max().item() for v in up), default=0.0)
    print(f"[validate] LoRA {os.path.basename(loras[-1])}: {len(sd)} tensors, {len(bad)} non-finite, "
          f"max |up| {moved:.3g}")
    if bad:
        fails.append(f"LoRA has {len(bad)} non-finite tensors")
    if moved == 0.0:
        fails.append("LoRA up-weights are all zero (nothing trained)")

imgs = sorted(glob.glob(f"{out_dir}/samples/*"))
if not imgs:
    fails.append("no sample generated")
else:
    a = np.asarray(Image.open(imgs[-1]).convert("RGB")).astype(float)
    print(f"[validate] sample with LoRA: mean {a.mean():.1f} std {a.std():.1f}")
    if a.std() < 10 or a.mean() < 8:
        fails.append("sample with LoRA is black/flat")

print("[validate] RESULT:", "PASS" if not fails else "FAIL - " + "; ".join(fails), flush=True)
sys.exit(1 if fails else 0)
