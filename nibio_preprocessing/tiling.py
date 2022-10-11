import argparse
import pdal
import os, glob
import json
import pandas as pd
from tqdm import tqdm

class Tiling:
    """
    The tiling operation on las files is done by the pdal splitter filter.
    """
    def __init__(self, input_folder, output_folder, tile_size=10, tile_buffer=0, do_mapping_to_ply=False, do_tile_index=False):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.output_folder_ply = output_folder + "_ply"
        self.tile_size = tile_size
        self.tile_buffer = tile_buffer
        self.do_mapping_to_ply = do_mapping_to_ply
        self.do_tile_index = do_tile_index
 

    def do_tiling_of_single_file(self, file):
        """
        This function will tile a single file into smaller files
        """

        # get a name for the file 
        file_name_base = os.path.basename(file)
        file_name = os.path.splitext(file_name_base)[0]

        # create a folder for the file
        file_folder = os.path.join(self.output_folder, file_name)
        if not os.path.exists(file_folder):
            os.makedirs(file_folder)

        # create a pipeline for the file
        data = {
            "pipeline":[
                { 
                    "filename":file,
                    #"spatialreference":"EPSG:25832" 
                },
                { 
                    "type":"filters.splitter", 
                    "length":str(self.tile_size), 
                    "buffer":str(self.tile_buffer) 
                },
                {
                    "type":"writers.ply",
                    "storage_mode":"little endian",
                    "filename":file_folder + "/#.ply" 
                  
                }
            ]
        }
        # do the pdal things
        json_string = json.dumps(data)
        pipeline = pdal.Pipeline(json_string)
        tile = pipeline.execute()

        # convert the tiles to ply
        if self.do_mapping_to_ply:
            self.rename_files_in_the_folder_and_extend_with_digits(file_folder)

        # get the tile index
        if self.do_tile_index:
            self.get_tile_index(file_folder)


    def do_tiling_of_files_in_folder(self):
        """
        This function will tile all the files in a folder
        """
        
        # check if the output folder exists and remove it
        if os.path.exists(self.output_folder):
            os.system("rm -r " + self.output_folder)

        os.makedirs(self.output_folder)

        # create a destination folder for all the tiles
        # if not os.path.exists(self.output_folder):
        #     os.makedirs(self.output_folder)

        # get all the files in the input folder (ply format assummed)
        files = glob.glob(self.input_folder + "/*.ply") 

        # loop through all the files
        for file in tqdm(files):
            self.do_tiling_of_single_file(file)

    def convert_single_file_from_las_to_ply(self, file):
        """
        This function will convert a single file from las to ply
        """

        # get a name for the file 
        file_name_base = os.path.splitext(file)[0]

        # create a pipeline for the file
        data = {
            "pipeline":[
                { # read input data
                    "type":"readers.ply",
                    "filename":file,
                    #"spatialreference":"EPSG:25832" 
                },
                {
                    "type":"writers.ply",
                    "filename":file_name_base +"#.ply" 
                }
            ]
        }
        # do the pdal things
        pipeline = pdal.Pipeline(json.dumps(data))
        pipeline.execute()

    def convert_files_in_folder_from_las_to_ply(self, folder=None):
        """
        This function will convert all the files in a folder from las to ply
        """

        # get all the files in the folder and subfolders las or laz
        files = glob.glob(folder + "/*.las") + glob.glob(folder + "/*.laz")

        # loop through all the files
        for file in tqdm(files):
            self.convert_single_file_from_las_to_ply(file)

    def get_tile_index(self, folder=None):
        """
        This function will create a tile index for all the tiles in a folder
        """
        tile_index = pd.DataFrame(columns=['tile', 'x', 'y'])
        files = glob.glob(os.path.join(folder, "*.ply"))

        for i, ply in tqdm(enumerate(files), total=len(files)):
            T = int(os.path.split(ply)[1].split('.')[0])
            reader = {"type":f"readers{os.path.splitext(ply)[1]}", "filename":ply}
            stats =  {"type":"filters.stats", "dimensions":"X,Y"}
            JSON = json.dumps([reader, stats])
            pipeline = pdal.Pipeline(JSON)
            pipeline.execute()
            X = pipeline.metadata['metadata']['filters.stats']['statistic'][0]['average']
            Y = pipeline.metadata['metadata']['filters.stats']['statistic'][1]['average']
            tile_index.loc[i, :] = [T, X, Y]   

        tile_index.to_csv(os.path.join(folder, 'tile_index.dat'), index=False, header=False, sep=' ')
    
    def remove_files_in_folder(self, folder=None, file_type=None):
        """
        This function will remove all the files in a folder of a certain type
        """
        files = glob.glob(folder + "/*." + file_type)
        for file in files:
            os.remove(file)

    def three_digits(self, number):
        """
        This function will add leading zeros to a number
        """
        if number < 10:
            return "00" + str(number)
        elif number < 100:
            return "0" + str(number)
        else:
            return str(number)

    def rename_files_in_the_folder_and_extend_with_digits(self, folder=None):
        """
        This function will rename all the files in a folder
        """
        files = glob.glob(folder + "/*.ply")
        for i, file in enumerate(files):
            os.rename(file, os.path.join(folder, self.three_digits(i) + ".ply"))

    def run(self):
        # self.convert_files_in_folder_from_las_to_ply(self.input_folder)
        self.do_tiling_of_files_in_folder()


def main(input_folder, output_folder, tile_size=10, tile_buffer=0, do_mapping_to_ply=False, do_tile_index=False):
    """
    This function will tile all the files in a folder
    """

    tiling = Tiling(input_folder, output_folder, tile_size, tile_buffer, do_mapping_to_ply, do_tile_index)
    tiling.run()


if __name__ == "__main__":
    # read command line arguments
    parser = argparse.ArgumentParser(description="Tiling")
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="Input folder containing las files",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        help="Output folder containing las files",
        required=True,
    )   
    parser.add_argument(
        "-t",
        "--tile_size",
        type=int,
        help="Tile size in meters",
        required=False,
        default=10,
    )
    parser.add_argument(
        "-b",
        "--tile_buffer",
        type=int,
        help="Tile buffer in meters",
        required=False,
        default=0,
    )
    parser.add_argument(
        "-m",
        "--do_mapping_to_ply",
        type=bool,
        help="Do mapping to ply",
        required=False,
        default=True,
    )
    parser.add_argument(
        "-g",
        "--do_tile_index",
        type=bool,
        help="Get tile index",
        required=False,
        default=True,
    )

    args = parser.parse_args()

    main(
        args.input_folder, 
        args.output_folder, 
        args.tile_size, 
        args.tile_buffer, 
        args.do_mapping_to_ply,
        args.do_tile_index
        )



 
