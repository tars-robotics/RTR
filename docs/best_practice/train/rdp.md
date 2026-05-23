# download our provided data, and put them in rtr_async_sys
```
data
`-- ckpts
    |-- peel_cucumber_15hz 
    |-- peel_cucumber_60hz 
    |-- wipe_vase_15hz 
    |-- wipe_vase_60hz 
    |-- write_board_15hz
    `-- write_board_60hz
```

# train_vae
```
conda activate rtr_rdp
cd third_party/reactive_diffusion_policy
# wandb login
bash scripts/train/rdp_vae/n_embed_10/train/peel_cucumber/train_rdp_vae_60hz_horizon48.sh
# after training, copy the ckpt to data/ckpts/peel_cucumber_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon48
```

# train_dp
```
cd third_party/reactive_diffusion_policy
# wandb login
bash scripts/train/dp_60hz/peel_cucumber/train_dp_ddim30_action60hz_image_down_sample_ratio4_horizon48.sh
# after training, copy the ckpt to data/ckpts/peel_cucumber_60hz/ckpts_abs/dp/horizon48
```


# train_dp_latent
```
cd third_party/reactive_diffusion_policy
# wandb login
bash scripts/train/rdp_ldp/peel_cucumber/train_rdp_ldp_60hz_horizon48.sh
# after training, copy the ckpt to data/ckpts/peel_cucumber_60hz/ckpts_abs/ldp/with_rdp_vae
```