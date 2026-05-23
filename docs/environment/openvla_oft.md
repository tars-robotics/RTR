create conda environment
```
conda create -n rtr_openvla_oft python=3.10 -y
conda activate rtr_openvla_oft
conda install ffmpeg -c conda-forge
```

using our modified openvla-oft
```
cd third_party/openvla_oft/openvla-oft
pip install -e .
```
or install the official openvla-oft
```
# TODO: we recommand using our modified openvla-oft, because modify official openvla-oft into RTR is complex, you can follow the following instruments in `third_party/openvla_oft/openvla_oft_finetune/readme.md`
```

link model wrappers and configs into openvla-oft
```
cd third_party/openvla_oft 
ln -s openvla_oft_inference/* openvla_oft/
ln -s openvla_oft_finetune/finetune* openvla_oft/
```

link model wrappers and configs into openvla-oft-latent
```
cd third_party/openvla_oft_latent
ln -s openvla_oft_inference/* openvla_oft/
ln -s openvla_oft_finetune/finetune* openvla_oft/
```

LIBERO
```
cd third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .
# if can't find libero in using, add `__init__.py` in `third_party/LIBERO/libero`
```

```
pip install torch==2.2.0 torchvision==0.17.0 
```

```
pip install  zarr scipy threadpoolctl numba
```

rtr_async_sys
```
cd rtr_async_sys
pip install -e .

opencv-python==4.11.0.86
```

bug_fix
```
pip install hydra-core==1.3.2 numpy==1.26.2 tokenizers==0.19.1 protobuf==4.21.12 transformers==4.40.1  wandb==0.23.1 einops==0.8.1 dill==0.4.0
```
