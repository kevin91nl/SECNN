import argparse
import csv
import os

import numpy as np
import pandas as pd

from preprocess import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the SECNN model on the given preprocessed input files.')
    parser.add_argument('input',
                        help='Path to the input files (folder containing preprocessed JSON files).')
    parser.add_argument('glove_file',
                        help='Path to the GloVe word embeddings file.')
    args = parser.parse_args()

    ########################
    # Load word embeddings #
    ########################

    # Load the GloVe file
    df_words = pd.read_table(args.glove_file, sep=' ', index_col=0, header=None, quoting=csv.QUOTE_NONE)

    # Load the vocabulary and the weight matrix
    VOCAB_WORDS = df_words.index.values.tolist()
    W_words = df_words.values
    words_count, words_dim = W_words.shape

    # Initialize vectors for the <PAD> (zeros) and <UNK> (initialized using Xavier initializer) markers
    xavier_bound = np.sqrt(6.) / np.sqrt(words_count + words_dim)
    markers_matrix = [np.zeros((1, words_dim))]
    markers_matrix += [np.random.uniform(-xavier_bound, xavier_bound, (1, words_dim))]
    markers_matrix = np.vstack(markers_matrix)

    # Augment the word vocabulary with the markers
    VOCAB_WORDS = ['<PAD>', '<UNK>'] + VOCAB_WORDS
    W_words = np.vstack([markers_matrix, W_words])

    #########################
    # Load train/test files #
    #########################

    files = os.listdir(args.input)
    for file in files[:1]:
        path = os.path.join(args.input, file)
        with open(path, 'r') as file_handle:
            data = json.load(file_handle)

            # Preprocess the data
            tokens = corenlp_to_tokens(data['nlp_data'])
            entities = get_entities(tokens)
            entities = align_entities(data['salient_entities'] + data['nonsalient_entities'], entities)
            for index, entity in enumerate(entities):
                entity[0]['is_salient'] = entity[0]['aligned_with'] in data['salient_entities'] if 'aligned_with' in \
                                                                                                   entity[0] else False

            # Only use aligned entities
            entities = [entity for entity in entities if len(entity) > 0 and 'aligned_with' in entity[0].keys()]
            entities = cluster_entities(entities)
            tokens, entities = replace_entities(tokens, entities)

            # Fetch all entity windows
            entity_windows = {entity[0]['label']: [] for entity in entities}
            for entity in entities:
                entity_windows[entity[0]['label']] = get_entity_windows(entity, tokens, replace_by_target=True)
