import argparse
import numpy as np
import laspy

class AddGroundToInstSeg():
        
    """Add ground to instance segmentation"""
    def __init__(self, sem_seg_file_path, inst_seg_file_path, output_file_path, verbose) -> None:
        self.sem_seg_file = sem_seg_file_path
        self.inst_seg_file = inst_seg_file_path
        self.output_file = output_file_path
        self.verbose = verbose

    def add_ground_to_inst_seg(self):
        # read sem seg
        sem_seg = laspy.read(self.sem_seg_file)
        # read inst seg
        inst_seg = laspy.read(self.inst_seg_file)
        # get point format of inst seg
        point_format = inst_seg.point_format
        if self.verbose:
            print("Point format of instance segmentation: {}".format(point_format.id))
        # get header of inst seg
        header = inst_seg.header
        if self.verbose:
            print("Header of instance segmentation: {}".format(header.version))

        # create new header
        new_header = laspy.LasHeader(point_format=point_format.id, version=header.version)

        # add extra dimension with label to the new header
        new_header.add_extra_dim(laspy.ExtraBytesParams(name='label', type=np.int32))
        new_header.add_extra_dim(laspy.ExtraBytesParams(name='instance_nr', type=np.int32))

        # create a new las file
        las = laspy.LasData(new_header)

        tmp_dict = {}
        small_subset = ['X', 'Y', 'Z', 'intensity', 'raw_classification', 'label','gps_time', 'red', 'green', 'blue', 'instance_nr']

        for item in small_subset:
            tmp_dict[item] = []

        for item in list(small_subset):
            tmp_dict[item] = np.append(tmp_dict[item], inst_seg[item])

        # add point of label 0 to the new file except for the instance_nr
        for item in small_subset:
            if item != 'instance_nr':
                tmp_dict[item] = np.append(tmp_dict[item], sem_seg[item][sem_seg.label == 0])

        # add instance_nr to the new file
        tmp_dict['instance_nr'] = tmp_dict['instance_nr'] + 1

        # append numpy vector of zeros to the instance_nr which is the same length as the points of label 0
        tmp_dict['instance_nr'] = np.append(
            tmp_dict['instance_nr'], 
            np.zeros(sem_seg['label'][sem_seg.label == 0].shape[0])
            )
        
        for key in tmp_dict.keys():
            las[key] = tmp_dict[key]

        las.write(self.output_file)

        # if verbose is selected write the name of the output file the results will be saved to
        if self.verbose:
            print("Results are saved to: {}".format(self.output_file))
      

        return las

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sem_seg_file", type=str, help="Path to semantic segmentation file")
    parser.add_argument("--inst_seg_file", type=str, help="Path to instance segmentation file")
    parser.add_argument("--output_file", type=str, help="Path to output file")
    parser.add_argument("--verbose", action='store_true', help="Print information about the process")
    args = parser.parse_args()

    add_ground_to_inst_seg = AddGroundToInstSeg(
        sem_seg_file_path=args.sem_seg_file,
        inst_seg_file_path=args.inst_seg_file,
        output_file_path=args.output_file,
        verbose=args.verbose
        )
    add_ground_to_inst_seg.add_ground_to_inst_seg()
