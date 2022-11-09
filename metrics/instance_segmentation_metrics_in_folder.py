import glob
import os
import laspy
from metrics.instance_segmentation_metrics import InstanceSegmentationMetrics

class InstanceSegmentationMetricsInFolder():
    def __init__(
        self,
        gt_las_folder_path,
        target_las_folder_path,
        remove_ground=False,
        verbose=False
    ):
        self.gt_las_folder_path = gt_las_folder_path
        self.target_las_folder_path = target_las_folder_path
        self.remove_ground = remove_ground
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

            # read the las file check if las file is not empty
            gt_las_file = laspy.read(gt_las_file_path)
            if len(gt_las_file.points) == 0:
                # remove the las file and the corresponding target_las_file_path
                os.remove(gt_las_file_path)
                os.remove(target_las_file_path)
                if self.verbose:
                    print('Removed empty las file: ' + gt_las_file_path)

            # check if las file is not empty
            target_las_file = laspy.read(target_las_file_path)
            if len(target_las_file.points) == 0:
                # remove the las file and the corresponding target_las_file_path
                os.remove(gt_las_file_path)
                os.remove(target_las_file_path)
                if self.verbose:
                    print('Removed empty las file: ' + target_las_file_path)

            # print what files are being matched and processed
            if self.verbose:
                print('Matching: ' + gt_las_file_path + ' and ' + target_las_file_path)


            # get the core name of the gt_las_file_path
            gt_las_file_core_name = os.path.basename(gt_las_file_path).split('.')[0]
            # get the core name of the target_las_file_path
            target_las_file_core_name = os.path.basename(target_las_file_path).split('.')[0]

            # check that the core name of the gt_las_file_path and target_las_file_path are the same
            if gt_las_file_core_name == target_las_file_core_name:
                # make a tuple of the matched paths
                matched_paths.append((gt_las_file_path, target_las_file_path)) 

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
                if self.verbose:
                    print('Processing: ' + gt_las_file_path + ' and ' + target_las_file_path)
        
                # run the instance segmentation metrics
                instance_segmentation_metrics = InstanceSegmentationMetrics(
                    gt_las_file_path,
                    target_las_file_path,
                    remove_ground=self.remove_ground,
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
    parser.add_argument('--remove_ground', action='store_true', help="Do not take into account the ground (class 0).", default=False)
    parser.add_argument('--verbose', action='store_true', help="Print information about the process")
    args = parser.parse_args()

    # run the instance segmentation metrics in folder
    instance_segmentation_metrics_in_folder = InstanceSegmentationMetricsInFolder(
        args.gt_las_folder_path,
        args.target_las_folder_path,
        args.remove_ground,
        verbose=args.verbose
    )

    mean_f1_score = instance_segmentation_metrics_in_folder.main()