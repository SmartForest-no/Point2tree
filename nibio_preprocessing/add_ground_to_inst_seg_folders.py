import argparse
import glob
import os
from nibio_preprocessing.add_ground_to_inst_seg import AddGroundToInstSeg


class AddGroundToInstSegFolders():
    """Add ground to instance segmentation"""
    def __init__(self, sem_seg_folder_path, inst_seg_folder_path, output_folder_path, verbose) -> None:
        self.sem_seg_folder = sem_seg_folder_path
        self.inst_seg_folder = inst_seg_folder_path
        self.output_folder = output_folder_path
        self.verbose = verbose

    def add_ground_to_inst_seg_folders(self):
        # get all files in the folder
        sem_seg_files = glob.glob(self.sem_seg_folder + '/*.las')
        inst_seg_files = glob.glob(self.inst_seg_folder + '/*.las')

        # print name of files
       
        if self.verbose:
            print("Sem seg files: {}".format(sem_seg_files))
            print("Inst seg files: {}".format(inst_seg_files))

        # check if same amount of files if not throw an exception and abort the programm
        if len(sem_seg_files) != len(inst_seg_files):
            print("The amount of files in the semantic segmentation folder and the instance segmentation folder is not the same. Please check the input folders.")
            
        # if output folder does not exist create it
        if not os.path.isdir(self.output_folder):
            os.mkdir(self.output_folder)

        # sort the files
        sem_seg_files.sort()
        inst_seg_files.sort()
        # match each sem seg file to a corresponding inst seg file with same core file name
        for sem_file in sem_seg_files:
            for inst_file in inst_seg_files:
                if os.path.basename(sem_file).split('.')[0] == os.path.basename(inst_file).split('.')[0]:
                    file_name = os.path.basename(inst_file)
                    # create output file path
                    output_file_path = os.path.join(self.output_folder, file_name)
            
                    # create instance of AddGroundToInstSeg
                    add_ground = AddGroundToInstSeg(
                        sem_file, inst_file, output_file_path , self.verbose)
                    # add ground to instance segmentation
                    add_ground.add_ground_to_inst_seg()

if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser(
        description='Add ground to instance segmentation')
    parser.add_argument('--sem_seg_folder', dest='sem_seg_folder',
                        type=str, help='Semantic segmentation folder', default='./data/')  
    parser.add_argument('--inst_seg_folder', dest='inst_seg_folder',
                        type=str, help='Instance segmentation folder', default='./data/')
    parser.add_argument('--output_folder', dest='output_folder',
                        type=str, help='Output folder', default='./output_folder/00')
    parser.add_argument('--verbose', action='store_true', help="Print information about the process")
    args = parser.parse_args()

    # create instance of AddGroundToInstSegFolders
    add_ground_to_inst_seg_folders = AddGroundToInstSegFolders(
        sem_seg_folder_path=args.sem_seg_folder,
        inst_seg_folder_path=args.inst_seg_folder,
        output_folder_path=args.output_folder,
        verbose=args.verbose
    )
    # add ground to instance segmentation
    add_ground_to_inst_seg_folders.add_ground_to_inst_seg_folders()



