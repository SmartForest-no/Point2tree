import sys
import hashlib

# taken from : https://www.geeksforgeeks.org/compare-two-files-using-hashing-in-python/
# and changed to fit my needs

def hashfile(file):

	# A arbitrary (but fixed) buffer
	# size (change accordingly)
	# 65536 = 65536 bytes = 64 kilobytes
	BUF_SIZE = 65536

	# Initializing the sha256() method
	sha256 = hashlib.sha256()

	# Opening the file provided as
	# the first commandline argument
	with open(file, 'rb') as f:
		
		while True:
			
			# reading data = BUF_SIZE from
			# the file and saving it in a
			# variable
			data = f.read(BUF_SIZE)

			# True if eof = 1
			if not data:
				break
	
			# Passing that data to that sh256 hash
			# function (updating the function with
			# that data)
			sha256.update(data)

	
	# sha256.hexdigest() hashes all the input
	# data passed to the sha256() via sha256.update()
	# Acts as a finalize method, after which
	# all the input data gets hashed hexdigest()
	# hashes the data, and returns the output
	# in hexadecimal format
	return sha256.hexdigest()

# Calling hashfile() function to obtain hashes
# of the files, and saving the result
# in a variable

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file1", help="first file to compare")
    parser.add_argument("file2", help="second file to compare")
    args = parser.parse_args()
    file1_hash = hashfile(args.file1)
    file2_hash = hashfile(args.file2)

    # Comparing the hashes
    if file1_hash == file2_hash:
        print("identical")
    else:
        print("different")
