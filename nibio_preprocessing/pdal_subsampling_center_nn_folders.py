import argparse
import glob
import json
import os
import pdal
from tqdm import tqdm

import laspy

from nibio_preprocessing.pdal_subsampling_center_nn import PDALSubsamplingCenterNN
from nibio_postprocessing.attach_labels_to_las_file_gt2pred import AttachLabelsToLasFileGt2Pred

class PDALSubsamplingCenterNNFolders:
    def __init__(self, input_folder, output_folder, voxel_size, verbose):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.voxel_size = voxel_size
        self.verbose = verbose

    def subsample_folders(self):
        # get all files in the folder
        input_files = glob.glob(self.input_folder + '/*.las')
        # print name of files
        if self.verbose:
            print("Input files: {}".format(input_files))

        # if output folder does not exist create it
        if not os.path.isdir(self.output_folder):
            os.mkdir(self.output_folder)

        # sort the files
        input_files.sort()
        # subsample each file
        for input_file in tqdm(input_files):
            file_name = os.path.basename(input_file)
            # create output file path
            output_file_path = os.path.join(self.output_folder, file_name)
            # create instance of PDALSubsamplingCenterNN
            subsample = PDALSubsamplingCenterNN(
                input_file, output_file_path, self.voxel_size)
            # subsample
            subsample.subsample()

    def transfer_extra_fields(self):
        # transfer extra fields from input to output
        # get all files in the folder
        input_files = glob.glob(self.input_folder + '/*.las')
        output_files = glob.glob(self.output_folder + '/*.las')
        # print name of files
        if self.verbose:
            print("Input files: {}".format(input_files))
            print("Output files: {}".format(output_files))
        
        # sort the files
        input_files.sort()
        output_files.sort()
        
        # check if the number of input and output files are the same
        if len(input_files) != len(output_files):
            raise Exception("Number of input and output files are not the same")

        # read input files and output files one by one using laspy
        for input_file, output_file in tqdm(zip(input_files, output_files)):
            # use AttachLabelsToLasFile to transfer extra fields
            transfer = AttachLabelsToLasFileGt2Pred(
                gt_las_file_path=input_file,
                target_las_file_path=output_file,
                update_las_file_path=output_file,
                gt_label_name='label',
                target_label_name='label',
                verbose=self.verbose
                )
            transfer.main()

            # transfer instance segmentation labels
            transfer = AttachLabelsToLasFileGt2Pred(
                gt_las_file_path=input_file,
                target_las_file_path=output_file,
                update_las_file_path=output_file,
                gt_label_name='treeID',
                target_label_name='treeID',
                verbose=self.verbose
                )
            transfer.main()

    def main(self):
        # subsample
        self.subsample_folders()
        # transfer extra fields
        self.transfer_extra_fields()

        if self.verbose:
            print("Done subsampling and transferring extra fields")

if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser(
        description='Subsample point cloud with PDAL')
    parser.add_argument('--input_folder', dest='input_folder',
                        type=str, help='Input folder', default='./data/')
    parser.add_argument('--output_folder', dest='output_folder',
                        type=str, help='Output folder', default='./output_folder')
    parser.add_argument('--voxel_size', dest='voxel_size',
                        type=float, help='Voxel size', default=0.02)
    parser.add_argument('--verbose', dest='verbose',
                        type=bool, help='Verbose', default=False)
    args = parser.parse_args()

    # create instance of PDALSubsamplingCenterNNFolders
    subsample_folders = PDALSubsamplingCenterNNFolders(
        args.input_folder, args.output_folder, args.voxel_size, True)
    # subsample
    subsample_folders.main()