How to Docker:
Build an image from a provided Dockerfile with
    docker build -f Dockerfile -t asuka .

Then run a built image with
    docker run -ti asuka:latest

Also able to mount a drive/folder with absolute paths and with gpu access via
    docker run --gpus 3 -v /path/to/folder:/root/data/ -ti asuka:latest
