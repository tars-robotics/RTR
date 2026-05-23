# download data and train vae
follow the instruments in `docs/best_practice/train/rdp.md` to download data and train vae

# process data
The downloaded data is in zarr-format, should convert it into lerobot v3 dataset format to train policies based on lerobot
```
cd third_party/lerobot/tools/convert_zarr_to_lerobot
bash scripts/peel_cucumber/xarm_peel_cucumber_state9_60hz.sh
bash scripts/peel_cucumber/xarm_peel_cucumber_state9_15hz.sh
bash scripts/peel_cucumber/xarm_peel_cucumber_state9_60hz_wo_img.sh # used to get statastics of VAE
```

# train pi05

```
conda activate rtr_lerobot
cd third_party/lerobot/lerobot
bash scripts/train/pi0.5/peel_cucumber/train_pi0.5_60hz.sh
```

# train p05-latent

## get statistics
```
cd third_party/lerobot/lerobot
bash scripts/get_statistics/vae/peel_cucumber/eval_rdp_vae_get_statistics_60hz.sh
# then copy the generated dataset_stats.json into `data/ckpts/peel_cucumber_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon48`
```

## train pi05-latent
```
cd third_party/lerobot/lerobot
bash scripts/train/pi0.5/peel_cucumber/train_pi0.5_latent_rdp_vae_60hz.sh
```
