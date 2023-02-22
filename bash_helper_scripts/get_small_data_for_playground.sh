TARGET_FOLDER=/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground
rm -rf $TARGET_FOLDER/*

cp /home/nibio/mutable-outside-world/data/small_file_pipeline_test/small_file_pipeline_test.las $TARGET_FOLDER

# change name of the file to first.laz 
mv $TARGET_FOLDER/small_file_pipeline_test.las $TARGET_FOLDER/first.las

# make a copy of the file
# cp $TARGET_FOLDER/first.las $TARGET_FOLDER/second.las

# # make a copy of the file
# cp $TARGET_FOLDER/first.las $TARGET_FOLDER/third.las

# # make a copy of the file
# cp $TARGET_FOLDER/first.las $TARGET_FOLDER/fourth.las