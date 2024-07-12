docker run \
    -it \
    --gpus all \
    --name boda-ai-server \
    --network host \
    -v .:/home/boda-ai \
    pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime \
    bash
