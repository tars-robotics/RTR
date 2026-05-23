create conda environment
```
conda create -y -n rtr_lerobot python=3.10
conda activate rtr_lerobot
conda install ffmpeg -c conda-forge
```

lerobot: using lerobot in our third-party
```
cd third_party/lerobot/lerobot
```
or install the official lerobot
```
cd third_party/lerobot
rm -rf lerobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git switch --detach v0.4.3
```

pip install lerobot
```
cd third_party/lerobot/lerbot
pip install -e .
pip install -e ".[pi]"
```

link model wrappers and configs into lerobot
```
cd third_party/lerobot 
ln -s *.py lerobot/
ln -s configs lerobot/
ln -s scripts lerobot/
```

rtr_async_sys
```
cd rtr_async_sys
pip install -e .
```

```
pip install  zarr scipy threadpoolctl numba tyro==1.0.3
```