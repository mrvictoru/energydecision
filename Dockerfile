FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

WORKDIR /code

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    tini

COPY ./requirements.txt ./

COPY ./torch_req.txt ./

RUN pip install --no-cache-dir -r torch_req.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x /usr/bin/tini

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]