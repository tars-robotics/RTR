# download data and train vae
follow the instruments in `docs/best_practice/train/rdp.md` to download data and train vae


# process data
rlds_env should be build before process data for openvla_oft, follow `docs/environment/rlds_env.md` to build it

```
conda activate rlds_env
cd third_party/openvla_oft_latent/openvla_oft_finetune/peel_cucumber_60hz_oft_6drotate_dataset
bash get_data.sh
```

# train openvla-oft
```
conda activate rtr_openvla_oft
cd third_party/openvla_oft/openvla-oft
bash myscripts/peel_cucumber/train/finetune_6d_rorate_60hz.sh
```

# train openvla-oft-latent
```
conda activate rtr_openvla_oft
cd third_party/openvla_oft_latent/openvla-oft
bash myscripts/peel_cucumber/train/finetune_6d_rotate_latent_with_rdp_vae.sh
```