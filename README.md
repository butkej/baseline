## How to Docker:
Build an image from a provided Dockerfile with
    ```docker build -f Dockerfile -t asuka .```

Then run a built image with
    ```docker run -ti asuka:latest```

Also able to mount a drive/folder with absolute paths and with gpu access via
    ```docker run --gpus 3 -v /path/to/folder:/root/data/ -ti asuka:latest```

## Dataset:
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
