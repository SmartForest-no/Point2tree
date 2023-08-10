#/bin/sh

# make mutable-outside-world dir if does not exits
DOCKER_LOCATION="/home/nibio/"
MUTABLE_OUTSIZE_WORLD_DIR="/home/nibio/mutable-outside-world"
mkdir -p $MUTABLE_OUTSIZE_WORLD_DIR

docker run --gpus all \
           --mount type=volume,src=nibio-cuda-vscode-data,dst=${DOCKER_LOCATION}/data \
           --mount type=volume,src=nibio-cuda-vscode-server,dst=${DOCKER_LOCATION}/.vscode-server \
           --mount type=volume,src=nibio-cuda-ssh,dst=${DOCKER_LOCATION}/.ssh \
           --mount type=bind,src=${MUTABLE_OUTSIZE_WORLD_DIR},dst=${DOCKER_LOCATION}/mutable-outside-world \
		       --name nibio-cuda-vscode-conda-v1 \
		       --shm-size 32GB \
		       --ipc private \
		       --publish 127.0.0.1::22 \
		       -itd nibio/cuda-vscode-conda:latest
