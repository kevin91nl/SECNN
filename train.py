import argparse
import csv
import os

import chainer
import numpy as np
import pandas as pd
from chainer import training
from chainer.datasets import TransformDataset, split_dataset
from chainer.iterators import SerialIterator
from chainer.training import extensions

from model.secnn import SECNN, SECNNLossWrapper
from preprocess import Preprocessor
from preprocess.files import JSONFileLoader
from preprocess.tokens import Tokenizer
from preprocess.vocab import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the SECNN model on the given preprocessed input files.')
    parser.add_argument('input',
                        help='Path to the input files (folder containing preprocessed JSON files).')
    parser.add_argument('glove_file',
                        help='Path to the GloVe word embeddings file.')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot.')
    parser.add_argument('--test_size', default=5, type=int,
                        help='Number of test documents used for validation.')
    args = parser.parse_args()

    df_words = pd.read_table(args.glove_file, sep=' ', index_col=0, header=None, quoting=csv.QUOTE_NONE)

    ####################################################################################################################

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

    # Convert vocab lists to dictionaries
    VOCAB_WORDS = {word: index for index, word in enumerate(VOCAB_WORDS)}
    VOCAB_POSTAGS = {postag: index for index, postag in enumerate(VOCAB_POSTAGS)}
    VOCAB_ENTITIES = {entity: index for index, entity in enumerate(VOCAB_ENTITIES)}

    ####################################################################################################################

    tokenizer = Tokenizer(vocab_words=VOCAB_WORDS, vocab_postags=VOCAB_POSTAGS, vocab_entities=VOCAB_ENTITIES)
    preprocessor = Preprocessor(tokenizer)

    files = [os.path.join(args.input, file) for file in os.listdir(args.input)]

    file_loader = JSONFileLoader(preprocessor)
    dataset = TransformDataset(files, file_loader.load_file)
    test_set, train_set = split_dataset(dataset, 1)

    train_iter = SerialIterator(train_set, batch_size=1, repeat=True, shuffle=True)
    test_iter = SerialIterator(test_set[:args.test_size], batch_size=args.test_size, repeat=False, shuffle=False)

    model = SECNN(
        config_word={'in_size': W_words.shape[0], 'out_size': W_words.shape[1], 'initialW': W_words},
        config_postag={'in_size': len(VOCAB_POSTAGS), 'out_size': 32},
        config_entity={'in_size': len(VOCAB_ENTITIES), 'out_size': 32},
        config_rnn={'in_size': None, 'out_size': 64},
        config_affine={'in_size': None, 'out_size': 1},
    )
    loss_model = SECNNLossWrapper(model)
    model.embed_word.disable_update()

    optimizer = chainer.optimizers.SGD()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer=optimizer, converter=lambda *arguments: arguments[0],
                                       loss_func=loss_model.__call__, device=-1)
    trainer = training.Trainer(updater, (1, 'epoch'), out='result')
    trainer.extend(extensions.Evaluator(test_iter, loss_model, converter=lambda *arguments: arguments[0]),
                   trigger=(5, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(5, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'loss']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.snapshot(), trigger=(100, 'iteration'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
