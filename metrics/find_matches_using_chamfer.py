import pdal
import json
import os

class FindMatchesUsingChamfer:
    def __init__(self, gt_folder, pred_folder, verbose=False):
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder

    def compare_gt_pred(self, las_gt, las_pred):
        # run pdal chamfer
        cmd = f"pdal chamfer {las_gt} {las_pred}"
        output = os.popen(cmd).read()

        # read the output as json
        output = json.loads(output)
        return output['chamfer']

if __name__ == '__main__':
    gt_file = '/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/results/input_data/output/0.las'
    pred_file = '/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/results/instance_segmented_point_clouds_with_ground/output/0.las'
    FindMatchesUsingChamfer('', '').compare_gt_pred(gt_file, pred_file)


