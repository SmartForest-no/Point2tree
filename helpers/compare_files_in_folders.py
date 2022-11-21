import os
from tqdm import tqdm
from helpers.compare_files_with_hash import hashfile

class CompareFilesInFolders:
    # compare files in two folders using hashfile
    def __init__(self, folder1, folder2, verbose=False):
        # read files in folder1
        self.files1 = [os.path.join(folder1, f) for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))]
        self.files2 = [os.path.join(folder2, f) for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))]
        self.files1.sort()
        self.files2.sort()
        self.verbose = verbose
        # cast exception if the number of files in the two folders are not equal
        if len(self.files1) != len(self.files2):
            raise Exception(f'number of files in {folder1} and {folder2} are not equal')

    def compare(self):
        # compare one by one using hashfile and count the number of matches
        matches = 0

        for i in tqdm(range(len(self.files1))):
            if hashfile(self.files1[i]) == hashfile(self.files2[i]):
                matches += 1
            else:
                if self.verbose:
                    print(f'files {self.files1[i]} and {self.files2[i]} are different')

        # return the number of errors and percentage of errors
        if self.verbose:
            print(f'files1: {len(self.files1)}')
            print(f'files2: {len(self.files2)}')
            print(f'matches: {matches}')
            print(f'errors: {len(self.files1) - matches}')
            print(f'percentage of matches: {matches / len(self.files1) * 100:.2f}%')
            print(f'percentage of errors: {100 - matches / len(self.files1) * 100:.2f}%')

        errors = len(self.files1) - matches
        procentage_of_errors = 100 - matches / len(self.files1) * 100
        return errors, procentage_of_errors

if __name__ == '__main__':
    # use argparse to parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder1', type=str, default='', help='folder 1')
    parser.add_argument('--folder2', type=str, default='', help='folder 2')
    parser.add_argument('--verbose', action='store_true', help='print stuff')
    args = parser.parse_args()

    # create an instance of CompareFilesInFolders
    c = CompareFilesInFolders(args.folder1, args.folder2, args.verbose)
    # compare the two folders
    c.compare()
    