
docker container rm test_oracle
docker image rm nibio/pc-geoslam-oracle
./build.sh
echo "Running the container"
# docker run --gpus all --name test_oracle nibio/pc-geoslam-oracle
docker run -it --gpus all \
    --name test_oracle \
    --mount type=bind,source=/home/nibio/mutable-outside-world/code/oracle_deploy/instance_segmentation_classic/local_folder_out/las_files,target=/app/local_folder_out/las_files \
    nibio/pc-geoslam-oracle



