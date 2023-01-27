

./bash_helper_scripts/get_austrian_sample_instance_p2.sh
pdal translate maciek/p2_instance.las maciek/p2_instance.ply
python fsct/run.py --model /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/fsct/model/model.pth --point-cloud maciek/p2_instance.ply --batch_size 10 --odir maciek/ --verbose --keep-npy

python helpers/compare_files_in_folders.py --folder1 maciek/p2_instance.tmp --folder2 old_maciek/p2_instance.tmp --verbose