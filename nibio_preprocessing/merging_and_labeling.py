# This is to merge the nibio data and label them in one cloud.

import os
import json
import pdal
import argparse
from tqdm import tqdm
import numpy as np
from plyfile import PlyData, PlyElement
import logging
logging.basicConfig(level=logging.DEBUG)


class LabelInstance(object):
    def __init__(self, ply_file, instance_nr):
        self.ply_file = ply_file
        self.instance_nr = instance_nr

    def read_ply(self):
        p = PlyData.read(self.ply_file)
        v = p.elements[0]
        return v

    def create_new_vertex_data(self, v):
        a = np.empty(len(v . data),  v.data.dtype.descr + [('instance_nr',   'i4')])
        for name in v.data.dtype.fields:
            a[name] = v[name]
        a['instance_nr'] = self.instance_nr
        return a

    def recreate_ply_element(self, a):
        v = PlyElement.describe(a, 'vertex')
        return v

    def recreate_ply_data(self, v):
        p = PlyData([v], text=True)
        return p

    def write_ply(self, p):
        p.write(self.ply_file)

    def label_instance(self):
        v = self.read_ply()
        a = self.create_new_vertex_data(v)
        v = self.recreate_ply_element(a)
        p = self.recreate_ply_data(v)
        self.write_ply(p)

# remove all the files in the folder and subfolders which contain the word "leafoff"
def remove_leafoff_files(data_folder):
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".ply"):
                if "leafoff" in file:
                    os.remove(os.path.join(root, file))


def merge_ply_files(data_folder, output_file='output_instance_segmented.ply'):
    """
        data_folder: the folder where the ply files are stored
    """
    logging.info("Merging the ply files")
    data = {"pipeline":[]}
    tags = []

    for root, _, files in os.walk(data_folder):
        for file in tqdm (files):
            if file.endswith(".ply"):
                data["pipeline"].append({"filename": os.path.abspath(os.path.join(root, file)), "tag": "tag_" + file.split(".")[0]})
                tags.append("tag_" + file.split(".")[0])

    data["pipeline"].append({"type": "filters.merge", "inputs": tags})

    if output_file == 'output_instance_segmented.ply':
        # if the output file is the default one, then we want to save it in the data folder
        data["pipeline"].append({"type":"writers.ply", "filename":os.path.join(data_folder, output_file)})
    else:
        # if the output file is not the default one, then we want to save it the target folder
        data["pipeline"].append({"type":"writers.ply", "filename":output_file})

    # print where the file is saved
    logging.info("The file is saved in: " + os.path.join(data_folder, output_file))
    logging.info("Merging was done for {} files".format(len(tags)))

    pipeline = pdal.Pipeline(json.dumps(data))
    pipeline.execute()
    
    logging.info("Done")


def main(data_folder, output_file="output_instance_segmented.ply"):
    """
        data_folder: the folder where the ply files are stored
    """
    logging.info("Removing all the files in the folder and subfolders which contain the word 'leafoff'")
    remove_leafoff_files(data_folder)

    if os.path.exists(os.path.join(data_folder, output_file)):
        logging.info(output_file)
        os.remove(os.path.join(data_folder, output_file))

    # get all the ply files in the folder
    instance_segmentation_id = 0

    logging.info("Labeling the ply files")

    for root, _, files in os.walk(data_folder):
        for file in tqdm (files):
            if file.endswith(".ply"):
                label_instance = LabelInstance(os.path.join(root, file), instance_segmentation_id)
                label_instance.label_instance()
                instance_segmentation_id += 1
    
    merge_ply_files(data_folder, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label the instances in the ply files.')
    parser.add_argument('--data_folder', type=str, help='The folder where the ply files are stored')
    parser.add_argument('--output_file', help='The output file name.', default="output_instance_segmented.ply")
    parser.add_argument('--only_merging', action='store_true', help='Only merge the ply files')

    args = parser.parse_args()

    if args.only_merging:
        print("Only merging the ply files")
        merge_ply_files(args.data_folder, args.output_file)
    else:
        print("Labeling instances in ply files and merging them afterwards")
        main(args.data_folder, args.output_file)



