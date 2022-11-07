import glob
import os
from metrics.instance_segmentation_metrics import InstanceSegmentationMetrics

class InstanceSegmentationMetricsInFolder():
    def __init__(
        self,
        gt_las_folder_path,
        target_las_folder_path,
        gt_label_name='gt_label', #TODO: implement the same as in the instance_segmentation_metrics.py
        target_label_name='target_label', #TODO: implement the same as in the instance_segmentation_metrics.py
        verbose=False
    ):
        self.gt_las_folder_path = gt_las_folder_path
        self.target_las_folder_path = target_las_folder_path
        self.gt_label_name = gt_label_name
        self.target_label_name = target_label_name
        self.verbose = verbose

    def main(self):
        # get all las files in the gt_las_folder_path
        gt_las_file_paths = glob.glob(self.gt_las_folder_path + '/*.las', recursive=False)
        gt_las_file_paths.sort()

        # get all las files in the target_las_folder_path
        target_las_file_paths = glob.glob(self.target_las_folder_path + '/*.las', recursive=False)
        target_las_file_paths.sort()

        # check that the number of las files in the gt_las_folder_path and target_las_folder_path are the same
        if len(gt_las_file_paths) != len(target_las_file_paths):
            raise Exception('The number of las files in the gt_las_folder_path and target_las_folder_path are not the same')

        # match the core name in the gt_las_file_path and target_las_file_path and make tuples of the matched paths
        matched_paths = []
        for gt_las_file_path, target_las_file_path in zip(gt_las_file_paths, target_las_file_paths):
            # get the core name of the gt_las_file_path
            gt_las_file_core_name = os.path.basename(gt_las_file_path).split('.')[0]
            # get the core name of the target_las_file_path
            target_las_file_core_name = os.path.basename(target_las_file_path).split('.')[0]

            # check that the core name of the gt_las_file_path and target_las_file_path are the same
            if gt_las_file_core_name == target_las_file_core_name:
                # make a tuple of the matched paths
                matched_paths.append((gt_las_file_path, target_las_file_path)) 
            print('matched_paths', matched_paths)

        # check if all are matched if not raise an exception
        if len(matched_paths) != len(gt_las_file_paths):
            raise Exception('Not all las files in the gt_las_folder_path and target_las_folder_path are matched')

        # run the instance segmentation metrics for each matched las file
        f1_scores = []

        for gt_las_file_path, target_las_file_path in matched_paths:
            # get the core name of the gt_las_file_path
            gt_las_file_core_name = os.path.basename(gt_las_file_path).split('.')[0]
            # get the core name of the target_las_file_path
            target_las_file_core_name = os.path.basename(target_las_file_path).split('.')[0]

            # check that the core name of the gt_las_file_path and target_las_file_path are the same
            if gt_las_file_core_name == target_las_file_core_name:
                # run the instance segmentation metrics
                instance_segmentation_metrics = InstanceSegmentationMetrics(
                    gt_las_file_path,
                    target_las_file_path,
                    verbose=self.verbose
                )
                _, f1_score_weighted = instance_segmentation_metrics.main()
                f1_scores.append(f1_score_weighted)

        # calculate the mean f1 score
        mean_f1_score = sum(f1_scores) / len(f1_scores)
        if self.verbose:
            print('Mean F1 Score: {}'.format(mean_f1_score))

        return mean_f1_score

if __name__ == '__main__':
    # arg parser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_las_folder_path', type=str, required=True)
    parser.add_argument('--target_las_folder_path', type=str, required=True)
    parser.add_argument('--gt_label_name', type=str, default='gt_label')
    parser.add_argument('--target_label_name', type=str, default='target_label')
    parser.add_argument('--verbose', action='store_true', help="Print information about the process")
    args = parser.parse_args()

    # run the instance segmentation metrics in folder
    instance_segmentation_metrics_in_folder = InstanceSegmentationMetricsInFolder(
        args.gt_las_folder_path,
        args.target_las_folder_path,
        gt_label_name=args.gt_label_name,
        target_label_name=args.target_label_name,
        verbose=args.verbose
    )

    mean_f1_score = instance_segmentation_metrics_in_folder.main()