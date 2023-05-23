
docker container rm test_oracle
docker image rm nibio/pc-geoslam-oracle
./build.sh
echo "Running the container"
# docker run --gpus all --name test_oracle nibio/pc-geoslam-oracle
docker run -it --gpus all --name test_oracle nibio/pc-geoslam-oracle


