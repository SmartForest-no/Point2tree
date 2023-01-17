import argparse
import laspy
import numpy as np


class DensityAndVolumeOfPc:
    '''
    This class is used to find the density of the pc
    '''

    def __init__(self, las_path)->None:
        self.las_path = las_path

    def __call__(self)->float:
        '''
        This function is used to find the density of the pc
        '''
        # load the point cloud
        pc = laspy.read(self.las_path)

        # get the points
        points = np.vstack((pc.x, pc.y, pc.z)).transpose()

        # get the bounding box
        bounding_box = np.array([points.min(axis=0), points.max(axis=0)])

        # get the volume of the bounding box
        volume = np.prod(bounding_box[1] - bounding_box[0])

        # get the number of points
        n_points = points.shape[0]

        # get the density
        density = n_points / volume

        # put desity and volume in a dictionary
        dict = {
            "density": density,
            "volume": volume
        }

        return dict,


if __name__ == "__main__":
    # use argparse to get the arguments
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--las_path", type=str, default="")
    args = parser.parse_args()

    # run the main function 
    density_and_volume_of_pc = DensityAndVolumeOfPc(
        las_path = args.las_path
    )

    # print desity and volume in a nice way
    out_dict = density_and_volume_of_pc()
    print("Density: " + str(out_dict[0]["density"]))
    print("Volume: " + str(out_dict[0]["volume"]))

