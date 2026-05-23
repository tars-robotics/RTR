# Download & Organize the Training Datasets

This guide explains:

1. The dataset layout that the training scripts expect (`data/ckpts/<task>/...`).
2. How to **download** our released training datasets from the Hugging Face
   Hub.
3. How to **unpack** them back into the layout expected by the scripts under
   `third_party/reactive_diffusion_policy/scripts/train/...`.

The released dataset lives at
[`sadpiggy/rtr_robot_sys_zarr`](https://huggingface.co/datasets/sadpiggy/rtr_robot_sys_zarr)
on the Hugging Face Hub.

---

## 1. What the training scripts expect

All training scripts under `third_party/reactive_diffusion_policy/scripts/train/`
read training data from `data/ckpts/<task>/`. Each task directory has the
following two subdirectories that are required for training:

```
data/
└── ckpts/
    ├── peel_cucumber_15hz/
    │   ├── rdp_zarr/
    │   │   └── replay_buffer.zarr/      # zarr group with episode data
    │   └── rdp_pca/
    │       ├── pca_matrix1.npy
    │       ├── pca_matrix2.npy
    │       ├── pca_mean1.npy
    │       └── pca_mean2.npy
    ├── peel_cucumber_60hz/               # same structure
    ├── wipe_vase_15hz/
    ├── wipe_vase_60hz/
    ├── write_board_15hz/
    └── write_board_60hz/
```

Other subdirectories that may appear under each task (e.g. `ckpts_abs/`,
`videos/`) are **outputs** produced by training/inference. They are not
distributed by this download bundle and you do not need them to start
training.

Task ↔ dataset sizes (uncompressed):

| Task                  | `rdp_zarr` size | `rdp_pca` size |
| --------------------- | --------------: | -------------: |
| `peel_cucumber_15hz`  |          3.7 GB |         < 1 MB |
| `peel_cucumber_60hz`  |           15 GB |         < 1 MB |
| `wipe_vase_15hz`      |          2.6 GB |         < 1 MB |
| `wipe_vase_60hz`      |          9.7 GB |         < 1 MB |
| `write_board_15hz`    |          3.2 GB |         < 1 MB |
| `write_board_60hz`    |           12 GB |         < 1 MB |
| **Total**             |     **~46 GB**  |       **~2 MB**|

Each task is shipped on the Hub as a **single `.tar` archive** that
contains both `rdp_zarr/` and `rdp_pca/`. We picked `.tar` (no
compression) because:

- The zarr replay buffers are already compressed at the chunk level, so
  `gzip` / `zstd` add cost with negligible savings.
- One large file per task is the most LFS-friendly upload pattern on
  Hugging Face, and downloads cleanly resume.

---

## 2. Download from Hugging Face

### Prerequisites

```bash
pip install -U "huggingface_hub[cli]"
# (optional) authenticate if the repo is private; otherwise this is not needed
huggingface-cli login
```

### Download all tasks

This downloads every `*.tar` from the repo into `data/zarr_dataset/`
(you can pick any local directory; the next step rearranges it):

```bash
cd /path/to/rtr_robot_sys

huggingface-cli download sadpiggy/rtr_robot_sys_zarr \
    --repo-type dataset \
    --local-dir data/zarr_dataset \
    --local-dir-use-symlinks False
```

After the command finishes you should see:

```
data/zarr_dataset/
├── peel_cucumber_15hz.tar
├── peel_cucumber_60hz.tar
├── wipe_vase_15hz.tar
├── wipe_vase_60hz.tar
├── write_board_15hz.tar
├── write_board_60hz.tar
└── README.md            # (the Hub repo's README, ignore)
```

### Download only the tasks you need

If disk is tight, you can fetch a single task by passing
`--include`:

```bash
huggingface-cli download sadpiggy/rtr_robot_sys_zarr \
    --repo-type dataset \
    --include "peel_cucumber_60hz.tar" \
    --local-dir data/zarr_dataset \
    --local-dir-use-symlinks False
```

### Python alternative (e.g. inside a notebook)

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sadpiggy/rtr_robot_sys_zarr",
    repo_type="dataset",
    local_dir="data/zarr_dataset",
    local_dir_use_symlinks=False,
    allow_patterns=["*.tar"],
)
```

---

## 3. Unpack into the layout expected by training

The training scripts read from `data/ckpts/<task>/`, so after downloading
into `data/zarr_dataset/` you need to extract each `.tar` into the
matching task directory under `data/ckpts/`.

### One-shot script

Run from the repo root:

```bash
mkdir -p data/ckpts

for tar in data/zarr_dataset/*.tar; do
    task=$(basename "$tar" .tar)
    echo "[extract] $tar -> data/ckpts/$task/"
    mkdir -p "data/ckpts/$task"
    tar -xf "$tar" -C "data/ckpts/$task"
done
```

Each archive contains exactly two top-level entries (`rdp_zarr/` and
`rdp_pca/`), so after extraction `data/ckpts/<task>/` will match the
layout in §1.

> If you already have other artifacts (e.g. `ckpts_abs/`) under
> `data/ckpts/<task>/`, the extraction will only add `rdp_zarr/` and
> `rdp_pca/` next to them and will not touch anything else.

### Verify

A quick sanity check:

```bash
python - <<'PY'
import zarr
from pathlib import Path

for task in sorted(Path("data/ckpts").iterdir()):
    zarr_root = task / "rdp_zarr" / "replay_buffer.zarr"
    if not zarr_root.exists():
        print(f"[missing] {zarr_root}")
        continue
    root = zarr.open(str(zarr_root), mode="r")
    n_episodes = root["meta/episode_ends"].shape[0]
    n_steps = int(root["meta/episode_ends"][-1]) if n_episodes else 0
    print(f"{task.name:24s}  episodes={n_episodes:4d}  steps={n_steps}")
PY
```

You should see a non-zero episode count for every task.

### Free up disk after extraction (optional)

The `.tar` files take another ~46 GB on top of the extracted dataset.
Once you've verified the extraction you can safely remove them:

```bash
rm -r data/zarr_dataset
```

---

## 4. Train

Once the data is in `data/ckpts/<task>/`, follow
[`docs/best_practice/train/rdp.md`](rdp.md) (or the per-policy guides
`lerobot.md` / `openvla_oft.md`) to launch training. As an example:

```bash
conda activate rtr_rdp
cd third_party/reactive_diffusion_policy
bash scripts/train/rdp_vae/n_embed_10/train/peel_cucumber/train_rdp_vae_60hz_horizon48.sh
# trained checkpoints land in data/ckpts/peel_cucumber_60hz/ckpts_abs/rdp_vae/...
```
