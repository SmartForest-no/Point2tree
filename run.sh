#!/bin/bash
docker run --mount type=bind,src='/home/opc/git_repos/instance_segmentation_classic/config/config.yaml',dst='/app/current_config.yaml' --rm nibio/cuda-vscode-conda:latest --path_to_config_file /app/current_config.yaml
