import os
import argparse

# check if every hdf5 file in dir has a corresponding artf file
def check_dataset(dir_path: str):
    hdf5_files = [f for f in os.listdir(dir_path) if f.endswith('.hdf5')]
    artf_files = [f for f in os.listdir(dir_path) if f.endswith('.artf')]

    for hdf5_file in hdf5_files:
        if hdf5_file.replace('.hdf5', '.artf') not in artf_files:
            print(f"Missing .artf file for {hdf5_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check if every hdf5 file in a directory has a corresponding artf file')
    parser.add_argument('dir_path', type=str, help='Path to directory containing hdf5 and artf files')
    args = parser.parse_args()

    check_dataset(args.dir_path)


