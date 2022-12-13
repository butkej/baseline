# Baseline for Histopathological Image Analysis
This repo is inspired by KatherLab's HIA repo found [here](https://github.com/KatherLab/HIA).  
It combines Multiple-instance learning based [CLAM](https://github.com/mahmoodlab/CLAM) and classical workflows (ResNet and ViT) for whole slide image classification.

[![Follow me on Twitter](https://img.shields.io/twitter/follow/JoshuaButke?style=social&logo=twitter)](https://twitter.com/intent/follow?screen_name=JoshuaButke)


## How to use:
Dockerfile as well as conda env is provided.
Run the script with
	python baseline.py -args
see ```src/utils.py``` for respective args

## How to Docker:
Build an image from a provided Dockerfile with  
    ```docker build -f Dockerfile -t asuka .```

Then run a built image with  
    ```docker run -ti asuka:latest```

Also able to mount a drive/folder with absolute paths and with gpu access via  
    ```docker run --gpus 3 -v /path/to/folder:/root/data/ -ti asuka:latest```

### Dataset:
Currently WSI data of 2422 slides (`/ml/wsi/`)  
On `/ml/slide_ID/` split into 3 slide_ID subgroups (classifier targets): DLBCL, FL, Reactive (more to come in the future, up to 10?)

Amounts:
- DLBCL 620
- FL 493
- Reactive 411

On `/ml/slide_ID_seperate` split into train and test sets

Amounts (train:test):
- DLBCL 496:124
- FL 394:99
- Reactive 328:83
