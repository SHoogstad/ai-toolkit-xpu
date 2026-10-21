# ai-toolkit (ostris) for Intel Arc (B580) via PyTorch XPU + a cuda->xpu shim.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates software-properties-common gpg-agent rsync \
      build-essential python3.12 python3-pip python3-dev python3-venv \
      ffmpeg libgl1 libglib2.0-0 tini procps \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update && apt-get install -y --no-install-recommends \
      libze-intel-gpu1 libze1 intel-opencl-icd intel-metrics-discovery clinfo \
    && curl -sL https://deb.nodesource.com/setup_23.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

ENV PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_CACHE_DIR=1
RUN pip install torch==2.13.0 torchvision==0.28.0 torchaudio \
      --index-url https://download.pytorch.org/whl/xpu

WORKDIR /app/ai-toolkit
ARG CACHEBUST=1
ARG GIT_COMMIT=main
RUN echo "$CACHEBUST" && git clone https://github.com/ostris/ai-toolkit.git . && git checkout ${GIT_COMMIT}

# deps; afterwards make sure nothing swapped the xpu torch for a cuda build
RUN pip install -r requirements.txt && pip install setuptools==69.5.1 && \
    python -c "import torch; v=torch.__version__; print('torch', v); assert 'xpu' in v, v"

# device-type checks: treat xpu like cuda
RUN grep -rlE "\.type ?[!=]= ?['\"]cuda['\"]" --include=*.py . | xargs -r sed -i -E \
      -e "s/\.type ?== ?['\"]cuda['\"]/.type in ('cuda', 'xpu')/g" \
      -e "s/\.type ?!= ?['\"]cuda['\"]/.type not in ('cuda', 'xpu')/g"

# B580: bitsandbytes' 8-bit AdamW is slow on XPU; plain AdamW was ~17% faster
# per step in an SDXL LoRA benchmark, so make it the new-job default
RUN sed -i "s/optimizer: 'adamw8bit'/optimizer: 'adamw'/" ui/src/app/jobs/new/jobConfig.ts && \
    grep -q "optimizer: 'adamw'," ui/src/app/jobs/new/jobConfig.ts

RUN cd ui && npm ci && npm run update_db && npm run build

# level-zero headers: triton-xpu JIT-compiles its driver against them
RUN apt-get update && apt-get install -y --no-install-recommends libze-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# loaded via .pth: Ubuntu ships its own sitecustomize.py that would shadow ours
COPY xpu_shim.py xpu_attention.py xpu_prefetch.py /usr/local/lib/python3.12/dist-packages/
RUN echo "import xpu_shim" > /usr/local/lib/python3.12/dist-packages/zz_xpu_shim.pth
COPY nvidia-smi /usr/local/bin/nvidia-smi
COPY start.sh /start.sh
COPY tests /opt/xpu-tests
RUN chmod +x /usr/local/bin/nvidia-smi /start.sh /opt/xpu-tests/run.sh

ENV AITK_XPU=1 \
    AI_TOOLKIT_DB_JOURNAL_MODE=DELETE \
    PYTORCH_ENABLE_XPU_FALLBACK=1 \
    NODE_ENV=production
EXPOSE 8675
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/start.sh"]
