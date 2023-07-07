import sys

def remove_prefix(lines):
    processed_lines = [line.split(' - ', 1)[1] if ' - ' in line else line for line in lines]
    return processed_lines

def process_file(input_filename, output_filename):
    with open(input_filename, 'r') as input_file:
        data = input_file.read()
    lines = data.split('\n')
    processed_lines = remove_prefix(lines)
    processed_data = '\n'.join(processed_lines)
    with open(output_filename, 'w') as output_file:
        output_file.write(processed_data)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Please provide both input and output filenames as command-line arguments.")
    else:
        input_filename = sys.argv[1]
        output_filename = sys.argv[2]
        process_file(input_filename, output_filename)
        print(f"Processed data saved to {output_filename}.")
