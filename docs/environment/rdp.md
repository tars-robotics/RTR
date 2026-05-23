create conda environemnt
```
conda create -n rtr_rdp python=3.9 -y
conda activate rtr_rdp
conda install ffmpeg -c conda-forge
```

install reactive_diffusion_policy
```
cd third_party/reactive_diffusion_policy
pip install -r requirements.txt
```

```
pip install torch==2.1.0 torchvision==0.16.0
pip install diffusers==0.30.3 transformers=4.40.1 peft==0.11.1
```

rtr_async_sys
```
cd rtr_async_sys
pip install -e .
```