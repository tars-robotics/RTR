install rlds_dataset
```
cd third_party
git clone https://github.com/kpertsch/rlds_dataset_builder.git
cd rlds_dataset_builder
conda env create -f environment_ubuntu.yml
```

install rtr_async_sys
```
cd ../
pip install -e .
pip install numpy==1.24.3 typing-extensions==4.5.0 zarr
```