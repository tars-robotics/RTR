# lerobot integration

This directory holds the integration overlay that lets `rtr_async_sys` drive
policies trained with the upstream
[lerobot](https://github.com/huggingface/lerobot) codebase (in particular
pi0.5 and its latent variants).

For environment setup and the end-to-end training pipeline, follow the
canonical guides:

- Install: `docs/environment/lerobot.md`
- Training pi0.5 / pi0.5-latent: `docs/best_practice/train/lerobot.md`

The notes below only document what lives inside this directory.

## Layout

```
configs/                       Hydra overlays mounted into lerobot/
scripts/                       Eval / training launchers mounted into lerobot/
pi0_5_model_wrapper.py         Synchronous pi0.5 wrapper
pi0_5_rtc_model_wrapper.py     RTC-style pi0.5 wrapper
pi0_5_latent_model_wrapper.py  Latent pi0.5 wrapper
pi0_5_latent_block_vae_model_wrapper.py
pi0_5_latent_rdp_vae_model_wrapper.py
pi0_5_latent_rdp_vae_rtc_model_wrapper.py
environment_lerobot.yml        Conda env file for the lerobot side
tools/                         Standalone helpers
```

## Checkpoint layout

lerobot reads checkpoints out of `./outputs` and `./outputs-vlta`. Point those
at wherever you store your trained weights, e.g.:

```
ln -s /path/to/your/ckpts/lerobot_pi0.5  ./outputs
ln -s /path/to/your/ckpts/pi05           ./outputs-vlta
```
