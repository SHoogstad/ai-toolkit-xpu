# ai-toolkit for Intel Arc (XPU)

A Docker image that runs [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) (UI and trainer) on an Intel Arc GPU. It was built and tested on an **Arc B580 (12 GB)**.

Upstream ai-toolkit only supports CUDA. Rather than patching every call site, this image keeps upstream unmodified (apart from two `sed` tweaks) and loads a small **cuda → xpu shim** when Python starts. The shim redirects `"cuda"` devices, `torch.cuda.*` and autocast to `torch.xpu`, and works around Intel driver memory behaviour (see [Memory on XPU](#memory-on-xpu)).

## What's in here

| File | Purpose |
|---|---|
| `Dockerfile` | Ubuntu 24.04, Intel compute runtime (kobuk-team PPA), PyTorch 2.13 XPU wheels, ai-toolkit + UI |
| `xpu_shim.py` | The cuda → xpu remapping, loaded through a `.pth` file when `AITK_XPU=1` |
| `xpu_attention.py` | Exact, memory-efficient attention for masked calls (see [Attention](#attention-on-xpu)) |
| `xpu_prefetch.py` | Background prefetch for ai-toolkit's layer offloading (see [Prefetch](#layer-offload-prefetch)) |
| `tests/` | Pass/fail tests for the XPU-specific parts, shipped in the image at `/opt/xpu-tests` (see [Tests](#tests)) |
| `nvidia-smi` | A stand-in so the UI can list the GPU and show VRAM, load, temperature and power (from DRM fdinfo and xe hwmon) |
| `start.sh` | Creates or upgrades the SQLite schema, then starts the UI |

## Get the image

A prebuilt image is on Docker Hub as [`hoogstaddie/ai-toolkit-xpu`](https://hub.docker.com/r/hoogstaddie/ai-toolkit-xpu):

```bash
docker pull hoogstaddie/ai-toolkit-xpu:latest
```

Tags: `latest` (the newest stable release), release tags such as `v1.2.0` (matching the [GitHub releases](https://github.com/SHoogstad/ai-toolkit-xpu/releases)), `beta` for the newest pre-release, and `0473387` for v1.0.0. It's about 12 GB on disk (4.1 GB download).

## Build

To build it yourself instead:

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
  --shm-size=16g --memory=28g \
  -e TZ=Europe/Amsterdam \
  -v /path/to/hf-cache:/root/.cache/huggingface/hub \
  -v /path/to/aitk_db.db:/app/ai-toolkit/aitk_db.db \
  -v /path/to/config:/app/ai-toolkit/config \
  -v /path/to/datasets:/app/ai-toolkit/datasets \
  -v /path/to/output:/app/ai-toolkit/output \
  hoogstaddie/ai-toolkit-xpu:latest
```

(Use `ai-toolkit:xpu` instead if you built it locally.)

Then open `http://<host>:8675`. Jobs can keep `device: cuda`, because the shim maps it to the Arc GPU.

> **Keep `AI_TOOLKIT_DB_JOURNAL_MODE=DELETE`** (the image sets it). The UI's default WAL mode keeps recent writes in an `aitk_db.db-wal` side file. When only the `.db` file is bind-mounted, that side file lives inside the container, and recreating the container loses recent jobs.

- **`--shm-size=16g`**: Docker's default 64 MB of shared memory is too small for the data-loader workers on large models. Krea 2 crashes with "unable to allocate shared memory". Only memory actually in use is consumed.
- **`--memory=…`** (adjust to your RAM): big models load their full checkpoint into RAM (Krea 2 peaks at about 26 GB). With a limit, a runaway job gets killed instead of pushing the host into swap until it stops responding.
- **Unraid:** mount the `.db` file from its disk path (`/mnt/diskN/…` or `/mnt/cache/…`), not `/mnt/user/…`. SQLite through the FUSE layer can fail mid-run with "attempt to write a readonly database" (error 1032). Job status then freezes, and Stop stops working.

## Benchmarks: Arc B580 vs RTX 3060 12 GB

SDXL LoRA (rank 16, 1024 px, gradient checkpointing on, bf16). Seconds per image, lower is better:

| Config | RTX 3060 | Arc B580 |
|---|---|---|
| AdamW8bit, batch 1 | 1.80 | 1.37 |
| AdamW, batch 1 | 1.73 | 1.17 |
| AdamW, batch 2 | 1.65 | **0.89** |

Raw bf16 matmul throughput: 27 TFLOPS on the 3060, 104 TFLOPS on the B580.

- **Use plain `adamw` on XPU.** bitsandbytes' 8-bit AdamW is slow there. The image makes `adamw` the UI default for new jobs.
- Keep **gradient checkpointing on**. Without it, SDXL at 1024 px doesn't fit in 12 GB: it runs out of VRAM (before v1.1.0 it silently spilled into system RAM at about 30 s/it).
- `torch.compile` does not work: inductor's `get_gpu_type` asserts when both `cuda` and `xpu` appear available.

## Verified models

- **SDXL**: LoRA training and sampling work.
- **Anima** (`circlestone-labs/Anima-Base-v1.0-Diffusers`): LoRA training and sampling work in bf16 and qfloat8. A 60-image character LoRA at batch 2 ran at about 2.4 s/img.
- **Krea 2** (`krea/Krea-2-Raw`, about 12B): LoRA training works on the 12 GB B580 with `convrot8` quantization, **60% transformer layer offloading**, batch 1 × gradient accumulation 2, and 512/768 px. Leave 1024 px out of the training resolutions. Seconds per step:

  | Version | Offload | Step time | OOM-skipped batches |
  |---|---|---|---|
  | v1.1.1 | 80% (60% runs out of VRAM) | about 35 s | about 1 in 10 |
  | **v1.2.0** | **60%** | **about 11–12 s** | **3 in 2000** |

  A full 2000-step character LoRA (60 images plus a regularization set) averaged 11.8 s/step including sample rounds, with 0 non-finite gradients and 98% of layer copies prefetched.

  v1.1.0 shipped a Krea 2 attention speed patch that **corrupted LoRAs**. It was removed in v1.1.1; see [Non-finite gradients](#non-finite-gradients).

Use [`tests/validate_model.py`](tests/validate_model.py) before trusting a new model: train about 30 steps, then confirm the saved LoRA is finite and a sample **with** the LoRA isn't black. A finite training loss doesn't prove the weights are fine. See [Tests](#tests).

## Memory on XPU

Intel's xe driver behaves differently from CUDA in two ways that matter for big models, and the shim compensates for both:

- **Pinned host memory doesn't scale.** The driver keeps only part of pinned RAM mapped for the GPU, and how much varies with system load: about 8 GB on an idle host, under 4 GB on a busy one. Past that, every host→GPU copy from pinned memory remaps it. Copies drop from 14 GB/s to about 0.5 GB/s, and the remap jobs can exceed the driver's job timeout, so the GPU resets with `UR_RESULT_ERROR_DEVICE_LOST`. Mixing pinned and pageable memory is worst of all. ai-toolkit's layer offloading pins everything, so **the shim makes `pin_memory()` a no-op by default** (`AITK_XPU_PIN_BUDGET_GB=0`). Pageable copies run at a steady 4–8 GB/s at any size.
- **VRAM is overcommitted silently.** The driver never refuses a GPU allocation. Past physical VRAM it evicts to system RAM, about 20× slower, so PyTorch's caching allocator keeps growing instead of reusing memory. **The shim caps PyTorch at 90% of VRAM** (`AITK_XPU_MEM_FRACTION=0.9`). PyTorch then reuses its cache, and a job that really doesn't fit gets a clear OOM instead of crawling.

Both can be overridden with `-e`. Set `AITK_XPU_MEM_FRACTION=1` for the old overcommitting behaviour.

## Non-finite gradients

A single inf/NaN gradient is enough to destroy a whole LoRA. Global-norm gradient clipping turns the total norm into NaN and multiplies it into **every** gradient, and the optimizer then writes NaN into all weights. The logged loss stays normal, and every sample afterwards decodes to black.

XPU kernels can hit this where CUDA doesn't. v1.1.0's Krea 2 patch let attention use XPU's fused SDPA kernel. Krea 2's first block has attention logits from about 3,000 to 31,000 (its keys share a large common component). On those, the fused kernel's backward is far off and overflowed to inf on the first step. v1.1.1 removed the patch; v1.2.0 handles these calls with an exact kernel instead (see [Attention](#attention-on-xpu)).

As a safety net for every model, **the shim checks gradients before each optimizer step**. If any are non-finite, it drops that step's update (weights and optimizer state stay untouched) and logs `[xpu_shim] non-finite gradients, skipping this optimizer step`. After 10 in a row (`AITK_XPU_NAN_GUARD_MAX`), it stops training with an error instead of producing a broken LoRA. `AITK_XPU_NAN_GUARD=0` disables it.

## Attention on XPU

On XPU, any attention mask sends `scaled_dot_product_attention` to PyTorch's MATH backend, which keeps the full fp32 score matrix and its gradient in VRAM: about 1.1 GB per tensor for Krea 2. XPU's fused kernel avoids that, but it can't take a mask, and its backward breaks down on extreme logits like Krea 2's first block.

The shim routes masked calls with a score matrix over `AITK_XPU_ATTN_MB` (512) through `xpu_attention.hybrid_attention`:

- **If the mask only hides a padded tail** (the usual text/alignment padding), the padding is sliced off. That's exactly equivalent, and the call runs unmasked.
- **If logits are moderate** (`max|q| · max|k| · √D ≤ AITK_XPU_FLASH_MAX`, default 2500), it uses the fused kernel. In that range its gradients stay at bf16-level error.
- **Otherwise**, it uses an exact chunked kernel. It processes queries in chunks and recomputes in the backward, never storing the full matrix (0.43 GB instead of 3.1 GB). Scores, softmax and P·V run in fp32, keys are centred, and the softmax is rebuilt from the saved row max and normalizer rather than a log-sum-exp. On Krea 2's real block-0 tensors its query gradient is 0.07% from a float64 reference, against 0.32% for MATH and over 600% for the fused kernel.

Smaller masked calls stay on MATH, and unmasked calls stay on the fused kernel. `AITK_XPU_ATTN_MB=0` disables the routing.

## Layer-offload prefetch

ai-toolkit's layer offloading copies each offloaded layer's weights to the GPU when that layer is called, on the training thread. With pageable memory that copy blocks, so copy and compute take turns. `xpu_prefetch` learns the order layers are called in from a 16-call history, which is long enough to tell the forward pass from gradient checkpointing's recomputation. It then copies the next `AITK_XPU_PREFETCH` (3) layers on a background thread and stream while the current one computes. After the first step it predicts every layer (Krea 2: 576 per step, 0 misses). Any mismatch, such as changed weights or an unexpected call, falls back to ai-toolkit's original path. `AITK_XPU_PREFETCH=0` disables it.

## Tests

```bash
docker exec ai-toolkit-xpu /opt/xpu-tests/run.sh
```

These cover the exact attention kernel against float64 (including the v1.1.0 failure mode), hybrid routing, the NaN guard, and the prefetcher (identical outputs, 0 misses after learning, stale-weight detection). Adding a job config also trains it end-to-end for 30 steps and checks the LoRA and a sample:

```bash
docker exec ai-toolkit-xpu /opt/xpu-tests/run.sh /app/ai-toolkit/output/<job>/.job_config.json
```

> **Don't run the tests while a training job uses the GPU.** Two processes together can exceed VRAM, and the driver then resets the GPU, killing the training job too.

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

## License

MIT, see [LICENSE](LICENSE). ai-toolkit itself is MIT-licensed by ostris.
