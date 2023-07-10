
docker container rm test_oracle
docker image rm nibio/pc-geoslam-oracle
./build.sh
echo "Running the container"
# docker run --gpus all --name test_oracle nibio/pc-geoslam-oracle
docker run -it --gpus all \
    --name test_oracle \
    --mount type=bind,source=/home/nibio/mutable-outside-world/code/oracle_deploy/instance_segmentation_classic/local_folder_oracle,target=/home/bucket/local_folder_oracle \
    --mount type=bind,source=/home/nibio/mutable-outside-world/code/oracle_deploy/instance_segmentation_classic/local_folder_oracle_hard,target=/home/bucket/local_folder_oracle_hard \
    nibio/pc-geoslam-oracle > test_oracle.log 2>&1



