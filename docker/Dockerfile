# Reference runtime for RoboWorld: Newton physics in Python 3.12 and Isaac Sim as render-only interpreters.
# Build from the repository root:  docker build -f docker/Dockerfile -t gym-anything-robotics .
# Run with an NVIDIA GPU that has RT cores (for example L40S or RTX 4090/5090; A100 and H100 cannot render):
#   docker run --gpus all -it -p 8080:8080 gym-anything-robotics
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates xz-utils unzip zstd build-essential pkg-config procps ffmpeg libatomic1 \
      libegl1 libgl1 libglu1-mesa libglx0 libgomp1 libsm6 libxi6 libxrandr2 libxt6 libglib2.0-0 libvulkan1 \
      libxkbcommon0 libxcursor1 libxinerama1 libfontconfig1 libfreetype6 libosmesa6 libegl-mesa0 libgl1-mesa-dri \
      libgles2 fonts-dejavu-core fonts-dejavu-extra \
 && apt-get purge -y mesa-vulkan-drivers || true \
 && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/0.9.5/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
 && uv python install 3.12.12 3.11

ARG RT=/opt/gar
# physics runtime
COPY requirements/newton.txt /tmp/newton.txt
RUN uv venv --python 3.12.12 $RT/newton15 && uv pip install --python $RT/newton15/bin/python -r /tmp/newton.txt
# render-only interpreters: Isaac Sim 5.1 (Python 3.11) and 6.0.1 (Python 3.12)
RUN uv venv --python 3.11 $RT/isaac51 \
 && uv pip install --python $RT/isaac51/bin/python torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128 \
      --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
 && uv pip install --python $RT/isaac51/bin/python 'isaacsim[all,extscache]==5.1.0.0' \
      --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease allow
RUN uv venv --python 3.12 $RT/isaac601 \
 && uv pip install --python $RT/isaac601/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 \
      --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
 && uv pip install --python $RT/isaac601/bin/python 'isaacsim[all,extscache]==6.0.1.0' \
      --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease allow
# OptiX denoiser library from the NVIDIA driver package, which headless containers do not mount
ARG DRIVER_URL=https://us.download.nvidia.com/XFree86/Linux-x86_64/610.57.04/NVIDIA-Linux-x86_64-610.57.04.run
ARG DRIVER_SHA256=b2e935c66b83bb00c0c857bc8e0ee0fd52de9286b40c9cc1eec29a7ce7eb116d
RUN curl -fsSL -o /tmp/drv.run $DRIVER_URL && echo "$DRIVER_SHA256  /tmp/drv.run" | sha256sum -c - \
 && sh /tmp/drv.run --extract-only --target /tmp/drv >/dev/null \
 && install -D -m 0644 /tmp/drv/nvoptix.bin /usr/share/nvidia/nvoptix.bin && rm -rf /tmp/drv /tmp/drv.run

COPY . /workspace/gym-anything-robotics
WORKDIR /workspace/gym-anything-robotics
RUN uv pip install --python $RT/newton15/bin/python -e ".[all]"

ENV PATH=$RT/newton15/bin:$PATH \
    GAR_WORKSPACE=/workspace/gym-anything-robotics \
    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
    NVIDIA_DRIVER_CAPABILITIES=all NVIDIA_VISIBLE_DEVICES=all \
    VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json PYGLET_HEADLESS=true \
    ISAAC51_PYTHON=$RT/isaac51/bin/python ISAAC601_PYTHON=$RT/isaac601/bin/python \
    ISAAC_RENDER_PYTHON=$RT/isaac51/bin/python FOOTBALL_RENDER_PYTHON=$RT/isaac51/bin/python \
    TITRATE_RENDER_PYTHON=$RT/isaac51/bin/python LAUNDRY_RENDER_PYTHON=$RT/isaac51/bin/python \
    LAUNDRY_RENDER_GRAPHICS_API=vulkan XDG_CACHE_HOME=$RT/cache MPLCONFIGDIR=$RT/cache/matplotlib
