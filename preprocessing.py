import argparse
import os
from tqdm import tqdm
import json

from preprocess import StanfordCoreNLPClient

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add the output of the Stanford CoreNLP pipeline and entity alignment information to the JSON '
                    'files found in the folder specified by the input argument.')
    parser.add_argument('input',
                        help='Path to the input files (folder containing JSON files).')
    parser.add_argument('--corenlp_url',
                        help='URL of the Stanford CoreNLP server.')
    parser.set_defaults(corenlp_url='http://localhost:9000')
    args = parser.parse_args()

    # Setup the Stanford CoreNLP client
    corenlp_client = StanfordCoreNLPClient(args.corenlp_url)

    # Process all the files
    progressbar = tqdm(os.listdir(args.input))
    for file in progressbar:
        progressbar.set_description(file)
        path = os.path.join(args.input, file)

        # Try to read the file
        file_data = None
        with open(path, 'r') as file_handle:
            file_data = json.load(file_handle)

        # Skip if the file could not be loaded
        if file_data is None:
            continue

        ####################################
        # Phase 1a: Enrich with NLP data.  #
        ####################################

        # Check for the nlp_data field
        if 'nlp_data' not in file_data:
            # Apply the Stanford CoreNLP pipeline to the text field
            file_text = file_data.get('text')
            nlp_data = corenlp_client(file_text)

            # Set the field
            file_data['nlp_data'] = nlp_data

        ####################################
        # Phase 1b: Simplify the NLP data. #
        ####################################

        ####################################
        # Phase 2: Align the entities.     #
        ####################################

        # Store the data
        with open(path, 'w') as file_handle:
            json.dump(file_data, file_handle)
