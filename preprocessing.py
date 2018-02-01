import argparse

import os
from tqdm import tqdm
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add the output of the Stanford CoreNLP pipeline and entity alignment information to the JSON '
                    'files found in the folder specified by the input argument.')
    parser.add_argument('input',
                        help='Path to the input files (folder containing JSON files).')
    parser.add_argument('--corenlp_url',
                        help='URL of the Stanford CoreNLP server.')
    parser.set_defaults(corenlp_url='http://localhost:9000/')
    args = parser.parse_args()

    # Test the Stanford CorenLP server
    progressbar = tqdm(os.listdir(args.input))
    for file in progressbar:
        progressbar.set_description(file)
        time.sleep(2)
