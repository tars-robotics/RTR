# openvla_oft (latent variant) - finetuning

For installation and the end-to-end training pipeline see the canonical guides:

- Install: `docs/environment/openvla_oft.md`
- Training (and dataset processing via the RLDS env):
  `docs/best_practice/train/openvla_oft.md` — the *latent* training section
  invokes the launchers in `myscripts/.../finetune_*latent_with_rdp_vae.sh`.

The base finetune setup (dataset registration, action / state conventions,
the `6drotate` flag, and the required edits to `prismatic/vla/constants.py`,
`prismatic/vla/datasets/rlds/oxe/configs.py`, and
`prismatic/vla/datasets/rlds/oxe/transforms.py`) is identical to the
non-latent variant — see
[`third_party/openvla_oft/openvla_oft_finetune/readme.md`](../../openvla_oft/openvla_oft_finetune/readme.md).

The notes below only cover what changes for **latent** training.

## Latent training
- Modify `openvla-oft/prismatic/vla/datasets/datasets.py` so that the VAE
  compresses the dataset actions into latents.
- Modify `openvla-oft/prismatic/util/data_utils.py` so the dataloader returns
  both `actions` and `original_actions`. `original_actions` is used to compute
  the L1 loss on the uncompressed actions.

### Normalization caveats (skip-normalization mode)
During training, the VLA's `predict_action` is not invoked, and unnormalization
only happens inside `predict_action`. The `predict_action` call inside
`finetune.sh` is the action-head version and does not unnormalize.

Inference (inferred): the dataset normalizes actions before feeding them in,
so the VLA learns on normalized actions. Losses are also computed on
normalized actions. At inference time (outside `finetune.py`), the predicted
normalized actions are then unnormalized to recover the real actions.

So VAE encoding must happen *before* normalization, and VAE decoding must
happen *after* unnormalization - otherwise the VAE sees inputs it has never
seen and cannot learn anything. A cleaner alternative is the normalized
variant below.

To run in this "skip-normalization" mode:
- Modify `normalize_action_and_proprio` in
  `openvla-oft/prismatic/vla/datasets/rlds/dataset.py` to skip normalization.
- Modify `openvla-oft/prismatic/extern/hf/modeling_prismatic.py` to skip
  unnormalization.

### Normalized training (VAE + openvla-oft)
- When training the VAE: encode after normalization, decode before
  unnormalization.
- When training latent OFT: the dataset normalizes the action, then the VAE
  encodes it to produce the latent (the OFT input); for the L1 loss, decode
  the latent back. At inference, the VAE finally unnormalizes.

### Latent normalize
After the VAE produces the latent and before it is fed into the VLM, it is
tokenized. Tokenization clips values outside [-1, 1], so the latent must be
normalized before being passed to the VLA.

The current dataset logic computes `dataset_statistics` from raw (unnormalized)
actions to obtain the action normalizer. We integrate `latent_normalize` into
the existing dataset, with the VAE consuming the normalized action. Flow:

```
read action
  -> normalize action
  -> VAE encode to obtain latent
  -> collect statistics over the latent
  -> obtain the latent normalizer
```

### Producing latent statistics
`vae_latent_dataset_statistics_path` is computed by running
`scripts/eval/pi0.5_vae/xarm_sponge_test1_60hz/dataset_eval/xxx/eval_pi0.5_vae/rdp_vae/eval_rdp_vae_get_statistics_60hz.sh`,
which writes `vae_latent_statistics`.
