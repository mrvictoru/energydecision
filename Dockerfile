FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

WORKDIR /code

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    tini \
    && rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 12.6 (stable, thoroughly tested; unlike CUDA 13 which is
# experimental — see pytorch.org/get-started/locally for supported CUDA versions).
# Installs all nvidia-*-cu12 stub libraries automatically as dependencies.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.7.1 torchvision==0.22.1

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x /usr/bin/tini

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]