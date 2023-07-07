# PCBEP
PCBEP is a B cell epitope predict tool which is based on esm-2 model and feature training consisting of atomic physicochemical properties, PSSM matrix.
## dataset 
link：https://pan.baidu.com/s/1BK4FWydiDZbZOxccEoSeOQ?pwd=zeiv 
Extract code：zeiv
## esm
###  Evolutionary Scale Modeling
link: https://github.com/facebookresearch/esm

As a prerequisite, you must have PyTorch installed to use this repository.
You can use this one-liner for installation, using the latest release of esm:
```
pip install fair-esm  # latest release, OR:
pip install git+https://github.com/facebookresearch/esm.git  # bleeding edge, current repo main branch
```
## Train model
```python train.py```
