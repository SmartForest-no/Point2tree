import json
import pdal


class PDALSubsamplingCenterNN:
    def __init__(self, input_file, output_file, voxel_size):
        self.input_file = input_file
        self.output_file = output_file
        self.voxel_size = voxel_size

    def subsample(self):
        pipeline = [
            {
                "type": "readers.las",
                "filename": self.input_file,
            },
            {
                "type": "filters.voxelcenternearestneighbor",
                "cell": self.voxel_size,
            },
            {
                "type": "writers.las",
                "filename": self.output_file,
                "extra_dims": "all"
            },
        ]

        pipeline = pdal.Pipeline(json.dumps(pipeline))
        pipeline.execute()

        return pipeline