#!/bin/sh
# This is a sample run file for the "hello" example.
# It is used to run the example from the main folder of the repository.

#### The first step of the pipeline: Do semantic segmentation on the whole point cloud (has to be .ply file) ####
# if the file is not .ply, convert it to .ply first using the follwing command:
# pdal translate sample_data/sample_point_cloud.laz sample_data/sample_point_cloud_mapped.ply

python fsct/run.py --point-cloud sample_data/sample_point_cloud.ply --batch_size 5 --odir sample_data

#### The second step of the pipeline: do the tiling and tile index generation ####
# move the output of the first step to the input folder of the second step

mkdir -p sample_data/segmented_point_clouds
mv sample_data/*.segmented.ply sample_data/segmented_point_clouds

python nibio_preprocessing/tiling.py -i sample_data/segmented_point_clouds/ -o sample_data/segmented_point_clouds/tiled

# This is the third step of the protol pipeline
# python3 fsct/points2trees.py -t  Plot82_2022-06-13_10-35-41_9pct_time_CLIPPED.segmented.ply --tindex /home/nibio/mutable-outside-world/data/raw_data/tiled_data/Plot82_2022-06-13_10-35-41_9pct_time_CLIPPED/tile_index.dat -o ../output_dir_our/ --n-tiles 5 --slice-thickness .5 --find-stems-height 2 --find-stems-thickness .5 --pandarallel --verbose --add-leaves --add-leaves-voxel-length .5 --graph-maximum-cumulative-gap 3 --save-diameter-class --ignore-missing-tiles

# python3 fsct/points2trees.py -t  /home/nibio/mutable-outside-world/data/raw_data/tiled_data/Plot82_2022-06-13_10-35-41_9pct_time_CLIPPED/000.ply --tindex /home/nibio/mutable-outside-world/data/raw_data/tiled_data/Plot82_2022-06-13_10-35-41_9pct_time_CLIPPED/tile_index.dat -o ../output_dir_our/ --n-tiles 5 --slice-thickness .5 --find-stems-height 2 --find-stems-thickness .5 --pandarallel --verbose --add-leaves --add-leaves-voxel-length .5 --graph-maximum-cumulative-gap 3 --save-diameter-class --ignore-missing-tiles

# python3 fsct/points2trees.py \
# -t /home/nibio/mutable-outside-world/data/segmented_data_conda/tiled_data/Plot82_2022-06-13_10-35-41_9pct_time_CLIPPED.segmented/000.ply \
# --tindex /home/nibio/mutable-outside-world/data/segmented_data_conda/tiled_data/Plot82_2022-06-13_10-35-41_9pct_time_CLIPPED.segmented/tile_index.dat \
# -o ../output_dir_our/ --n-tiles 3 --slice-thickness .5 --find-stems-height 2 --find-stems-thickness .5 --pandarallel --verbose --add-leaves --add-leaves-voxel-length .5 --graph-maximum-cumulative-gap 3 --save-diameter-class --ignore-missing-tiles

mkdir -p sample_data/instance_segmented_point_clouds

python3 fsct/points2trees.py \
-t sample_data/segmented_point_clouds/tiled/sample_point_cloud.segmented/000.ply \
--tindex sample_data/segmented_point_clouds/tiled/sample_point_cloud.segmented/tile_index.dat \
-o sample_data/instance_segmented_point_clouds/ \
--n-tiles 3 \
--slice-thickness .5 \
--find-stems-height 2 \
--find-stems-thickness .5 \
--pandarallel --verbose \
--add-leaves \
--add-leaves-voxel-length .5 \
--graph-maximum-cumulative-gap 3 \
--save-diameter-class \
--ignore-missing-tiles 