import logging
import laspy
import numpy as np


logging.basicConfig(level=logging.INFO)

class ReduceLabelsValuesInLas:
    def __init__(self, las_file_path, label_name, verbose=False):
        self.las_file_path = las_file_path
        self.label_name = label_name
        self.verbose = verbose

    def reduce_labels_values(self):
        # read las file
        las = laspy.read(self.las_file_path)

        point_format = las.point_format
        # get header of target las file
        header = las.header

        new_header = laspy.LasHeader(point_format=point_format.id, version=header.version)

        target_extra_dimensions = list(las.point_format.extra_dimension_names)

        # print all the target extra dimensions
        if self.verbose:
            logging.info('target_extra_dimensions: {}'.format(target_extra_dimensions))

        # add extra dimensions to new las file
        for item in target_extra_dimensions:
            new_header.add_extra_dim(laspy.ExtraBytesParams(name=item, type=np.int32))

        if 'height_above_DTM' in target_extra_dimensions:
            new_header.add_extra_dim(laspy.ExtraBytesParams(name='n_z', type=np.int32)) 

        new_las = laspy.LasData(new_header)
         # copy x, y, z, gt_label and target_label from target las file to the new las file
        new_las.x = las.x
        new_las.y = las.y
        new_las.z = las.z

        # copy contents of extra dimensions from target las file to the new las file
        for item in target_extra_dimensions:
            new_las[item] = las[item]

        new_las[self.label_name] = las[self.label_name] - 1
        # the line below is need because Phil has a different notation than Sean
        if 'height_above_DTM' in target_extra_dimensions:
            new_las['n_z'] = las['height_above_DTM']
            
        # write the new las file
        new_las.write(self.las_file_path)

    def main(self):
        self.reduce_labels_values(
        )
        if self.verbose:
            # write a report using logging
            logging.info('las_file_path: {}'.format(self.las_file_path))
            logging.info('label_name: {}'.format(self.label_name))

            # print the size of the las files
            las = laspy.read(self.las_file_path)
            las_size = las.x.shape[0]
            logging.info('las_size: {}'.format(las_size))



if __name__ == '__main__':
    # use parser to get arguments
    import argparse
    parser = argparse.ArgumentParser('Reduce labels values in las file')
    parser.add_argument('--las_file_path', type=str, required=True)
    parser.add_argument('--label_name', type=str, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # create an instance of the class
    reduce_labels_values_in_las = ReduceLabelsValuesInLas(
        las_file_path=args.las_file_path,
        label_name=args.label_name,
        verbose=args.verbose
    )

    # run the main function
    reduce_labels_values_in_las.main()
