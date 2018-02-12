import csv

import numpy as np
import pandas as pd

VOCAB_POSTAGS = ['<PAD>', '<UNK>', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS',
                 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD',
                 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

VOCAB_ENTITIES = ['<PAD>'] + ['@target'] + ['@entity%d' % i for i in range(1, 128)]


def load_glove_file(path):
    """Load the GloVe file.

    Parameters
    ----------
    path : str
        Path to the GloVe file.

    Returns
    -------
    list
        The word vocabulary (ordered list of words found in the GloVe file).
    np.matrix
        A n x m matrix (where n is the number of words in the vocabulary and m is the dimensionality of the vectors)
        containing the pre-trained weights found in the GloVe file.
    """

    df_words = pd.read_table(path, sep=' ', index_col=0, header=None, quoting=csv.QUOTE_NONE)

    # Load the vocabulary and the weight matrix
    vocab = df_words.index.values.tolist()
    weights = df_words.values
    words_count, words_dim = weights.shape

    # Initialize vectors for the <PAD> (zeros) and <UNK> (initialized using Xavier initializer) markers
    xavier_bound = np.sqrt(6.) / np.sqrt(words_count + words_dim)
    markers_matrix = [np.zeros((1, words_dim))]
    markers_matrix += [np.random.uniform(-xavier_bound, xavier_bound, (1, words_dim))]
    markers_matrix = np.vstack(markers_matrix)

    # Augment the word vocabulary with the markers
    vocab = ['<PAD>', '<UNK>'] + vocab
    weights = np.vstack([markers_matrix, weights])

    return vocab, weights
