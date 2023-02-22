import os
from collections import Counter
import laspy


def main (folder_path):
    # Define the path to the folder containing the LAS files
    # folder_path = "/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground"

    # Define the class codes you want to count
    class_codes = [0, 1, 2, 3, 4]

    # Define a dictionary to hold the counts for each class
    class_counts = {class_code: 0 for class_code in class_codes}

    # Iterate through the files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".las"):
            # Open the LAS file using laspy
            las_file = laspy.read(os.path.join(folder_path, file_name))

            # Count the points in each class and update the class_counts dictionary
            point_classes = Counter(las_file.label)
            for class_code in class_codes:
                class_counts[class_code] += point_classes[class_code]

    # Define the names of the classes
    names = ["ignore", "terrain", "vegetation", "CWD ", "stem"]

    # Print the class counts
    print("Class counts:")
    for class_code in class_codes:
        print(f"Class {names[class_code]}: {class_counts[class_code]}")

    # print it in percentages
    print("Class counts in percentages:")
    for class_code in class_codes:
        print(f"Class {names[class_code]}: {class_counts[class_code] / sum(class_counts.values()) * 100:.2f} %")


if __name__ == "__main__":
    # use argparse to parse the command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", help="Path to the folder containing the LAS files")

    args = parser.parse_args()

    main(args.folder_path)


