# Toolbx Guide for energydecision

[Toolbx](https://containertoolbx.org/) (formerly Toolbox) is a Red Hat / Fedora project
([source](https://github.com/containers/toolbox)) that wraps **Podman** to give you a
comfortable, interactive developer shell that lives inside an OCI container — without any
of the friction of `docker exec` or volume-mount gymnastics.

---

## 1 · How Toolbx Works

| Concept | Detail |
|---|---|
| **Engine** | Podman — rootless, daemonless. No `sudo`, no background service. |
| **Container image** | Any OCI image (Docker Hub, local build, etc.). |
| **Home directory** | Your real `$HOME` is bind-mounted automatically — dotfiles, SSH keys, git config, all present. |
| **Other mounts** | `/tmp`, `/media`, `/run/media`, `/run/host` (full host FS read-only), Wayland/X11 sockets, D-Bus, USB devices. |
| **Network** | Host network namespace — no port mapping needed. |
| **User identity** | Same UID inside as outside. Files you create are owned by you. |
| **Entry point** | `toolbox enter <name>` — drops into a login shell. Feels exactly like your normal terminal. |
| **Security posture** | Equivalent to a normal login session; not hardened for production isolation. |

The big ergonomic win: once you are inside a Toolbx shell your editor, IDE (VS Code, PyCharm,
Cursor, etc.), and the container's Python environment all see files at the same host paths —
no mount-mapping required.

---

## 2 · Why This Repo's Docker Setup Is Cumbersome

The current workflow (`docker-compose.yml`) requires:

```
# start
docker compose up -d

# open a shell
docker exec -it test_energy_container /bin/bash

# run training (note the path shift: workdir is /code/src)
python3 pretrain_decision_transformer.py ...

# stop
docker compose down
```

With Toolbx you just:

```
toolbox enter energydecision
cd ~/path/to/energydecision
python3 src/pretrain_decision_transformer.py ...
```

---

## 3 · Prerequisites

### 3a · Linux only

Toolbx is Linux-only (it uses Linux namespaces under the hood).  
On **Fedora** it is pre-installed. On **Ubuntu 22.04+**:

```bash
sudo apt install podman podman-toolbox
# Ubuntu packages the binary as 'toolbox' inside the 'podman-toolbox' package
```

On **Arch Linux**:

```bash
sudo pacman -S toolbox
```

### 3b · GPU passthrough (NVIDIA)

The current `docker-compose.yml` uses Docker's `--gpus all` shorthand.
Toolbx / Podman uses **CDI** (Container Device Interface) instead.

```bash
# Install NVIDIA Container Toolkit (if not already done for Docker)
sudo apt install -y nvidia-container-toolkit

# Generate the CDI spec (tells Podman where the GPU devices are)
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Verify Podman can see the GPU
podman run --rm --device nvidia.com/gpu=all \
  docker.io/nvidia/cuda:13.0.2-runtime-ubuntu22.04 nvidia-smi
```

---

## 4 · Setting Up the energydecision Toolbx Container

### Step 1 — Build the image with Podman

```bash
cd ~/path/to/energydecision

# Podman reads the same Dockerfile as Docker
podman build -t energydecision:latest .
```

> The `Dockerfile` uses `nvidia/cuda:13.0.2-runtime-ubuntu22.04` as the base and installs
> all Python dependencies from `requirements.txt` and `torch_req.txt`.

### Step 2 — Create the Toolbx container

**Without GPU** (CPU-only work, testing, notebooks without CUDA):

```bash
toolbox create --image energydecision:latest energydecision
```

**With GPU** (DT training, CUDA inference):

```bash
# Extra Podman flags are passed after '--'
toolbox create --image energydecision:latest -- --device nvidia.com/gpu=all energydecision-gpu
```

> Toolbx ≥ 0.0.99 supports passing extra `podman create` flags after `--`.

### Step 3 — Enter the container

```bash
# CPU container
toolbox enter energydecision

# GPU container
toolbox enter energydecision-gpu
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

Because Toolbx uses host networking, no port mapping is needed:

```bash
# inside the toolbox shell
cd ~/path/to/energydecision
jupyter notebook --port=8888 --no-browser --ip=127.0.0.1
```

Open `http://localhost:8888` in your host browser — it works immediately.

### Step 5 — Run the test suite

```bash
# inside the toolbox shell
cd ~/path/to/energydecision
python -m pytest tests/ -v
```

---

## 5 · Comparison: Docker Compose vs Toolbx

| Task | Docker Compose (current) | Toolbx |
|---|---|---|
| Enter a shell | `docker exec -it test_energy_container /bin/bash` | `toolbox enter energydecision` |
| Mount code | `-v .:/code` in compose file | Automatic (home dir is shared) |
| Working directory | `/code/src` (set in compose) | Your real repo path |
| Port mapping for Jupyter | `8888:8888` in compose | None (host network) |
| GPU | compose `deploy.resources` | `--device nvidia.com/gpu=all` via CDI |
| Root required | Docker daemon (root) | No — rootless Podman |
| Edit files with your IDE | Needs container-aware IDE config | Just open files normally |
| Stop container | `docker compose down` | Just `exit` the shell |

---

## 6 · Useful Toolbx Commands

```bash
# List all toolbx containers
toolbox list

# Enter a container
toolbox enter energydecision

# Run a single command without entering
toolbox run --container energydecision python3 src/pretrain_decision_transformer.py --help

# Remove a container (image stays)
toolbox rm energydecision

# Remove the image
podman rmi energydecision:latest
```

---

## 7 · Keeping Docker Compose for CI / Others

The existing `Dockerfile` and `docker-compose.yml` are unchanged and still work for:

- CI/CD pipelines that use Docker
- Team members on non-Linux systems (macOS / Windows with Docker Desktop)
- Production isolation scenarios

Toolbx is an **additional local development option**, not a replacement for the shared Docker
setup. Both can coexist on the same machine without conflict.

---

## 8 · Further Reading

- Toolbx project: <https://github.com/containers/toolbox>
- Podman docs: <https://docs.podman.io/>
- CDI spec (GPU passthrough): <https://github.com/cncf-tags/container-device-interface>
- NVIDIA Container Toolkit: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>
