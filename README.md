# ai-toolkit for Intel Arc (XPU)

A Docker image that runs [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) (UI and trainer) on an Intel Arc GPU. It was built and tested on an **Arc B580 (12 GB)**.

Upstream ai-toolkit only supports CUDA. Rather than patching every call site, this image keeps upstream unmodified (apart from two `sed` tweaks) and loads a small **cuda → xpu shim** when Python starts. The shim redirects `"cuda"` devices, `torch.cuda.*` and autocast to `torch.xpu`.

## What's in here

| File | Purpose |
|---|---|
| `Dockerfile` | Ubuntu 24.04, Intel compute runtime (kobuk-team PPA), PyTorch 2.13 XPU wheels, ai-toolkit + UI |
| `xpu_shim.py` | The cuda → xpu remapping, loaded through a `.pth` file when `AITK_XPU=1` |
| `nvidia-smi` | A stand-in so the UI can list the GPU and show VRAM, load, temperature and power (from DRM fdinfo and xe hwmon) |
| `start.sh` | Creates or upgrades the SQLite schema, then starts the UI |

## Build

```bash
docker build --build-arg CACHEBUST=$(date +%s) -t ai-toolkit:xpu .
```

- `CACHEBUST` forces a fresh `git clone` of ai-toolkit. Pin a commit with `--build-arg GIT_COMMIT=<sha>`.
- The build checks that `pip install -r requirements.txt` did not replace the XPU torch with a CUDA build.

## Run

```bash
touch /path/to/aitk_db.db
docker run -d --name ai-toolkit-xpu \
  --device /dev/dri/card0 --device /dev/dri/renderD128 \
  -p 8675:8675 \
  -e TZ=Europe/Amsterdam \
  -v /path/to/hf-cache:/root/.cache/huggingface/hub \
  -v /path/to/aitk_db.db:/app/ai-toolkit/aitk_db.db \
  -v /path/to/config:/app/ai-toolkit/config \
  -v /path/to/datasets:/app/ai-toolkit/datasets \
  -v /path/to/output:/app/ai-toolkit/output \
  ai-toolkit:xpu
```

Then open `http://<host>:8675`. Jobs can keep `device: cuda`, because the shim maps it to the Arc GPU.

> **Keep `AI_TOOLKIT_DB_JOURNAL_MODE=DELETE`** (the image sets it). The UI's default WAL mode keeps recent writes in an `aitk_db.db-wal` side file. When only the `.db` file is bind-mounted, that side file lives inside the container, and recreating the container loses recent jobs.

## Benchmarks: Arc B580 vs RTX 3060 12 GB

SDXL LoRA (rank 16, 1024 px, gradient checkpointing on, bf16). Seconds per image, lower is better:

| Config | RTX 3060 | Arc B580 |
|---|---|---|
| AdamW8bit, batch 1 | 1.80 | 1.37 |
| AdamW, batch 1 | 1.73 | 1.17 |
| AdamW, batch 2 | 1.65 | **0.89** |

Raw bf16 matmul throughput: 27 TFLOPS on the 3060, 104 TFLOPS on the B580.

- **Use plain `adamw` on XPU.** bitsandbytes' 8-bit AdamW is slow there. The image makes `adamw` the UI default for new jobs.
- Keep **gradient checkpointing on**. Without it, the B580 silently spills into system RAM (about 30 s/it).
- `torch.compile` does not work: inductor's `get_gpu_type` asserts when both `cuda` and `xpu` appear available.

## Verified models

- **SDXL**: LoRA training and sampling work.
- **Anima** (`circlestone-labs/Anima-Base-v1.0-Diffusers`): LoRA training and sampling work in bf16 and qfloat8. A 60-image character LoRA at batch 2 ran at about 2.4 s/img.

## Shim notes / gotchas

These are the problems that came up, recorded so they don't have to be rediscovered:

- **bitsandbytes is imported before `torch.cuda` is aliased.** It picks its backend at import time.
- **`torch.utils._device._device_constructors()` is called before the factory functions are wrapped.** Otherwise `with torch.device("meta")` no longer redirects them. transformers then leaves the non-persistent rotary `inv_freq` buffers uninitialized, Qwen text encoders (Anima, Krea 2) output NaN, and samples come out black.
- **The torch.cuda aliases are distinct wrapper functions.** torch._dynamo keys its handlers by function identity and asserts when `cuda.X is xpu.X`.
- **The wrapped factory functions are registered as TorchScript builtins**, so scripted code still resolves them to the aten ops.
- **`get_device_properties` / `get_device_capability` report Ampere (8.0)**, so code takes the generic bf16 paths.
- **The shim is loaded through `zz_xpu_shim.pth`, not `sitecustomize.py`.** Ubuntu ships its own sitecustomize, which would shadow ours.
- **`libze-dev` is required** because triton-xpu compiles its driver against the level-zero headers.

If something fails only on XPU, check the shim first.

## Credits

All the actual training code is [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit). This repo only packages it for Intel Arc.
