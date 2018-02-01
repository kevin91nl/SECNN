import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        help='Path to the input files (folder containing JSON files).')
    parser.add_argument('--corenlp_url', default='http://localhost:9000/',
                        help='URL of the Stanford CoreNLP server.')
    args = parser.parse_args()
    print(args)
