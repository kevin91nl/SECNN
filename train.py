import argparse
import os

import chainer
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
    # Load the arguments
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

    # Convert vocab lists to dictionaries
    VOCAB_WORDS, W_words = load_glove_file(args.glove_file)
    VOCAB_WORDS = {word: index for index, word in enumerate(VOCAB_WORDS)}
    VOCAB_POSTAGS = {postag: index for index, postag in enumerate(VOCAB_POSTAGS)}
    VOCAB_ENTITIES = {entity: index for index, entity in enumerate(VOCAB_ENTITIES)}

    # Create the tokenizer and the preprocessor
    tokenizer = Tokenizer(vocab_words=VOCAB_WORDS, vocab_postags=VOCAB_POSTAGS, vocab_entities=VOCAB_ENTITIES)
    preprocessor = Preprocessor(tokenizer)

    files = [os.path.join(args.input, file) for file in os.listdir(args.input)]

    # Create file loaders and transformations
    file_loader = JSONFileLoader(preprocessor)
    dataset = TransformDataset(files, file_loader.load_file)

    # Split the dataset and initialize the dataset iterators
    test_set, train_set = split_dataset(dataset, args.test_size)
    train_iter = SerialIterator(train_set, batch_size=1, repeat=True, shuffle=True)
    test_iter = SerialIterator(test_set[:args.test_size], batch_size=args.test_size, repeat=False, shuffle=False)

    # Initialize the model
    model = SECNN(
        config_word={'in_size': W_words.shape[0], 'out_size': W_words.shape[1], 'initialW': W_words},
        config_postag={'in_size': len(VOCAB_POSTAGS), 'out_size': 32},
        config_entity={'in_size': len(VOCAB_ENTITIES), 'out_size': 32},
        config_rnn={'in_size': None, 'out_size': 64},
        config_affine={'in_size': None, 'out_size': 1},
    )
    loss_model = SECNNLossWrapper(model)
    model.embed_word.disable_update()

    # Setup the optimizer
    optimizer = chainer.optimizers.SGD()
    optimizer.setup(model)

    # Create the updater and trainer
    updater = training.StandardUpdater(train_iter, optimizer=optimizer, converter=lambda *arguments: arguments[0],
                                       loss_func=loss_model.__call__, device=-1)
    trainer = training.Trainer(updater, (1, 'epoch'), out='result')
    trainer.extend(extensions.Evaluator(test_iter, loss_model, converter=lambda *arguments: arguments[0]),
                   trigger=(5, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(5, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'loss']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.snapshot(), trigger=(100, 'iteration'))

    # Resume from a specified snapshot
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the trainer
    trainer.run()
