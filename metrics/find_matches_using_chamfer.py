import pdal
import json
import os
import argparse

class FindMatchesUsingChamfer:
    def __init__(self, gt_folder, pred_folder, verbose=False):
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder
        self.verbose = verbose

    def compare_gt_pred(self, las_gt, las_pred):
        # run pdal chamfer
        cmd = f"pdal chamfer {las_gt} {las_pred}"
        output = os.popen(cmd).read()

        # read the output as json
        output = json.loads(output)
        return output['chamfer']

    def run_in_folders(self):
        # get all las files in gt and pred folder
        gt_files = [os.path.join(self.gt_folder, file) for file in os.listdir(self.gt_folder) if file.endswith(".las")]
        pred_files = [os.path.join(self.pred_folder, file) for file in os.listdir(self.pred_folder) if file.endswith(".las")]

        # compare all gt and pred files
        # define a dictionary to store the results
        results = {}
        for gt_file in gt_files:
            for pred_file in pred_files:
                print(gt_file, pred_file)
                print(self.compare_gt_pred(gt_file, pred_file))
                results[(gt_file, pred_file)] = self.compare_gt_pred(gt_file, pred_file)

        # sort the results in ascending order
        results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}

        # print the first 10 results
        for i, result in enumerate(results):
            if i < 10:
                print(result, results[result])
            else:
                break

        # save dictionary as csv using pandas
        import pandas as pd
        df = pd.DataFrame.from_dict(results, orient='index')
        df.to_csv('results_chamfer.csv')

if __name__ == '__main__':
    # use argparse to parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_folder', type=str, required=True, help='Path to the ground truth folder.')
    parser.add_argument('--pred_folder', type=str, required=True, help='Path to the predicted folder.')
    parser.add_argument('--verbose', type=bool, default=False, help='Print the output of pdal chamfer.')
    args = parser.parse_args()

    # run the class
    find_matches = FindMatchesUsingChamfer(args.gt_folder, args.pred_folder, args.verbose)
    find_matches.run_in_folders()






    # gt_file = '/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/results/input_data/output/0.las'
    # pred_file = '/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/results/instance_segmented_point_clouds_with_ground/output/0.las'
    # FindMatchesUsingChamfer('', '').compare_gt_pred(gt_file, pred_file)


