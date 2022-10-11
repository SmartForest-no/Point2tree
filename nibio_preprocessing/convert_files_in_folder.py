import os
import argparse
from matplotlib import use
from tqdm import tqdm

# usie logging to print out the progress
import logging
logging.basicConfig(level=logging.INFO)


class ConvertFilesInFolder(object):
    def __init__(self, input_folder, output_folder, out_file_type, verbose=False):
        """
        There are following available file types:
        - las
        - laz
        - ply
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.out_file_type = out_file_type
        self.verbose = verbose

    def convert_file(self, file_path):
        """
        Convert a single file to the specified output file type.
        """
        # if the output folder doesn't exist, create it
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # get the file name
        file_name = os.path.basename(file_path)
        # get the file name without extension
        file_name_no_ext = os.path.splitext(file_name)[0]
        # define the output file path
        output_file_path = os.path.join(self.output_folder, file_name_no_ext + "." + self.out_file_type)
        # define the command
        command = "pdal translate -i {} -o {}".format(file_path, output_file_path)
        # run the command
        os.system(command)

        # use logging to print out the progress
        logging.info("Converted file {} to {}.".format(file_name, self.out_file_type))

    # convert all files in the input folder
    def convert_files(self):
        """
        Convert all files in the input folder to the specified output file type.
        """
        # get paths to all files in the input folder and subfolders
        if self.verbose:
            print("Searching for files in the input folder...")

        file_paths = []
        for root, dirs, files in os.walk(self.input_folder):
            for file in files:
                file_paths.append(os.path.join(root, file))
        if self.verbose:
            # use logging to print out the progress
            logging.info("Found {} files.".format(len(file_paths)))

        # skip all the files that are not las or laz or ply
        file_paths = [f for f in file_paths if f.endswith(".las") or f.endswith(".laz") or f.endswith(".ply")]

        # skip all the files which are of type self.out_file_type
        file_paths = [f for f in file_paths if not f.endswith(self.out_file_type)]

        if self.verbose:
            # use logging to print out the progress
            logging.info("Found {} files that can be converted.".format(len(file_paths)))

        # iterate over all files and convert them
        for file_path in tqdm(file_paths):
            self.convert_file(file_path)
        
        # print out the progress
        if self.verbose:
            # use logging to print out the progress
            logging.info("Converted all files in the input folder to {}.".format(self.out_file_type))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", help="Path to the folder with the files to convert.")
    parser.add_argument("--output_folder", help="Path to the folder where the converted files will be saved.")
    parser.add_argument(
        "--out_file_type", 
        default='ply', 
        help="The file type of the output files.There are following available file types: las, laz, ply"
        )
    parser.add_argument("--verbose", help="Print more information.", action="store_true")
    args = parser.parse_args()
    # create an instance of the class
    converter = ConvertFilesInFolder(
        args.input_folder, 
        args.output_folder, 
        args.out_file_type, 
        args.verbose
        )
    # convert all files in the input folder
    converter.convert_files()
    
      