# Distrobox Guide for energydecision

[Distrobox](https://distrobox.it/) ([source](https://github.com/89luca89/distrobox)) is a
thin shell script that wraps **Podman or Docker** to give you a comfortable, interactive
developer shell that lives inside an OCI container — without any of the friction of
`docker exec` or volume-mount gymnastics.

---

## 1 · How Distrobox Works

| Concept | Detail |
|---|---|
| **Engine** | Podman (rootless/daemonless) **or** Docker — your choice. |
| **Container image** | Any OCI image (Docker Hub, local build, etc.). |
| **Home directory** | Your real `$HOME` is bind-mounted automatically — dotfiles, SSH keys, git config, all present. |
| **Other mounts** | `/tmp`, `/media`, `/run/media`, Wayland/X11 sockets, D-Bus, USB devices, audio. |
| **Network** | Host network namespace — no port mapping needed. |
| **User identity** | Same UID inside as outside. Files you create are owned by you. |
| **Entry point** | `distrobox enter <name>` — drops into a login shell. Feels exactly like your normal terminal. |
| **Export** | `distrobox-export` can expose container binaries/apps to the host desktop. |
| **Security posture** | Equivalent to a normal login session; not hardened for production isolation. |

The big ergonomic win: once you are inside a Distrobox shell your editor, IDE (VS Code,
PyCharm, Cursor, etc.), and the container's Python environment all see files at the same
host paths — no mount-mapping required.

---

## 2 · Why This Repo's Docker Setup Is Cumbersome

The current workflow (`docker-compose.yml`) requires:

```bash
# start
docker compose up -d

# open a shell
docker exec -it test_energy_container /bin/bash

# run training (note the path shift: workdir is /code/src)
python3 pretrain_decision_transformer.py ...

# stop
docker compose down
```

With Distrobox you just:

```bash
distrobox enter energydecision
cd ~/path/to/energydecision
python3 src/pretrain_decision_transformer.py ...
```

---

## 3 · Prerequisites

### 3a · Linux only

Distrobox is Linux-only (it uses Linux namespaces under the hood).  
It works on virtually any distro — Fedora, Ubuntu, Arch, Debian, openSUSE, etc.

**Recommended install (any distro, no root required):**

```bash
curl -s https://raw.githubusercontent.com/89luca89/distrobox/main/install | sh
# installs to ~/.local/bin — add that to your PATH if needed
```

**Or via package manager:**

```bash
# Fedora
sudo dnf install distrobox

# Ubuntu 23.04+
sudo apt install distrobox

# Arch Linux
sudo pacman -S distrobox
```

You also need **Podman or Docker** installed. Podman (rootless) is recommended:

```bash
# Ubuntu
sudo apt install podman

# Fedora (usually pre-installed)
sudo dnf install podman
```

### 3b · GPU passthrough (NVIDIA)

Distrobox supports two approaches depending on your container engine:

**With Docker (uses `--gpus all`):**

```bash
# Install NVIDIA Container Toolkit (if not already done)
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**With Podman (uses CDI):**

```bash
# Install NVIDIA Container Toolkit
sudo apt install -y nvidia-container-toolkit

# Generate the CDI spec
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Verify
podman run --rm --device nvidia.com/gpu=all \
  docker.io/nvidia/cuda:13.0.2-runtime-ubuntu22.04 nvidia-smi
```

---

## 4 · Setting Up the energydecision Distrobox Container

### Step 1 — Build the image

```bash
cd ~/path/to/energydecision

# Works with either Docker or Podman
docker build -t energydecision:latest .
# or
podman build -t energydecision:latest .
```

> The `Dockerfile` uses `nvidia/cuda:13.0.2-runtime-ubuntu22.04` as the base and installs
> all Python dependencies from `requirements.txt` and `torch_req.txt`.

### Step 2 — Create the Distrobox container

**Without GPU** (CPU-only work, testing, notebooks without CUDA):

```bash
distrobox create --name energydecision --image energydecision:latest
```

**With GPU via Docker (`--gpus all`):**

```bash
distrobox create \
  --name energydecision-gpu \
  --image energydecision:latest \
  --additional-flags "--gpus all"
```

**With GPU via Podman (CDI):**

```bash
distrobox create \
  --name energydecision-gpu \
  --image energydecision:latest \
  --additional-flags "--device nvidia.com/gpu=all"
```

### Step 3 — Enter the container

```bash
# CPU container
distrobox enter energydecision

# GPU container
distrobox enter energydecision-gpu
```

You are now in a shell inside the container. Your home directory is mounted, so:

```bash
cd ~/path/to/energydecision
python3 src/pretrain_decision_transformer.py \
  --data-dir data/household/logs \
  --epochs 2 --batch-size 6 --lr 2e-5 \
  --save-path models/household/dt/dt_model.pt \
  --checkpoint-path models/household/dt/dt_model_checkpoint.pt \
  --loss-csv-path models/household/dt/dt_model_loss_history.csv
```

Note: you are at the **repo root**, not `/code/src` as in the Docker Compose setup, so
scripts are referenced with a `src/` prefix and relative data/model paths work as-is.

### Step 4 — Run Jupyter Notebook

Because Distrobox uses host networking, no port mapping is needed:

```bash
# inside the distrobox shell
cd ~/path/to/energydecision
jupyter notebook --port=8888 --no-browser --ip=127.0.0.1
```

Open `http://localhost:8888` in your host browser — it works immediately.

### Step 5 — Run the test suite

```bash
# inside the distrobox shell
cd ~/path/to/energydecision
python -m pytest tests/ -v
```

---

## 5 · Comparison: Docker Compose vs Distrobox

| Task | Docker Compose (current) | Distrobox |
|---|---|---|
| Enter a shell | `docker exec -it test_energy_container /bin/bash` | `distrobox enter energydecision` |
| Mount code | `-v .:/code` in compose file | Automatic (home dir is shared) |
| Working directory | `/code/src` (set in compose) | Your real repo path |
| Port mapping for Jupyter | `8888:8888` in compose | None (host network) |
| GPU (Docker) | compose `deploy.resources` | `--additional-flags "--gpus all"` |
| GPU (Podman) | N/A | `--additional-flags "--device nvidia.com/gpu=all"` |
| Root required | Docker daemon (root) | No — rootless Podman supported |
| Edit files with your IDE | Needs container-aware IDE config | Just open files normally |
| Stop container | `docker compose down` | Just `exit` the shell |
| Works on any Linux distro | Yes (Docker) | Yes (Docker or Podman) |

---

## 6 · Useful Distrobox Commands

```bash
# List all distrobox containers
distrobox list

# Enter a container
distrobox enter energydecision

# Run a single command without entering (non-interactive)
distrobox enter energydecision -- python3 src/pretrain_decision_transformer.py --help

# Stop a running container
distrobox stop energydecision

# Remove a container (image stays)
distrobox rm energydecision

# Remove the image (Docker)
docker rmi energydecision:latest
# or (Podman)
podman rmi energydecision:latest
```

---

## 7 · Keeping Docker Compose for CI / Others

The existing `Dockerfile` and `docker-compose.yml` are unchanged and still work for:

- CI/CD pipelines that use Docker
- Team members on non-Linux systems (macOS / Windows with Docker Desktop)
- Production isolation scenarios

Distrobox is an **additional local development option**, not a replacement for the shared
Docker setup. Both can coexist on the same machine without conflict.

---

## 8 · Further Reading

- Distrobox project: <https://github.com/89luca89/distrobox>
- Distrobox documentation: <https://distrobox.it/>
- GPU inside Distrobox (NVIDIA): <https://github.com/89luca89/distrobox/blob/main/docs/useful_tips.md#using-nvidia-container-toolkit>
- Podman docs: <https://docs.podman.io/>
- NVIDIA Container Toolkit: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>
