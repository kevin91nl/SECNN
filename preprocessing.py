import argparse
import os

import requests
from tqdm import tqdm
import json
from urllib.parse import urlencode


class StanfordCoreNLPClient:

    def __init__(self, corenlp_base_url):
        self.session = requests.Session()
        self.corenlp_base_url = corenlp_base_url

    def __call__(self, text):
        query = {
            "properties": {
                "annotators": "tokenize,ssplit,pos,ner,coref",
                "timeout": 50000,
            },
            "pipelineLanguage": "en"
        }
        url = '%s/?%s' % (self.corenlp_base_url, urlencode(query))
        response = self.session.post(url, text)
        return json.loads(response.text)


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

        # Skip if the file is not loaded
        if file_data is None:
            continue

        # Skip if the NLP data is already set
        if 'nlp_data' in file_data:
            continue

        # Apply the Stanford CoreNLP pipeline
        file_text = file_data.get('text')
        nlp_data = corenlp_client(file_text)

        # Store the data
        file_data['nlp_data'] = nlp_data
        with open(path, 'w') as file_handle:
            json.dump(file_data, file_handle)
