#/bin/sh

docker build -t nibio/cuda-vscode-conda:latest --build-arg UID=`id -u` --build-arg GID=`id -g` --build-arg USERNAME=nibio .
