import numpy as np
import unidecode


class Tokenizer:
    """Converts (word, postag) tuples to [word_id, postag_id, entity_id] tuples.

    Example:
        >>> # Tokenize two sentences/windows: "hello world @entity" and "hello @entity":
        >>> tokenizer = Tokenizer(vocab_words={'<PAD>': 0, '<UNK>': 1, 'hello': 2, 'world': 2},
        >>>                       vocab_postags={'<PAD>': 0, '<UNK>': 1},
        >>>                       vocab_entities={'<PAD>': 0, '<UNK>': 1, '@entity': 2})
        >>>
        >>> tokenizer.tokenize_document({
        >>> '@entity': [
        >>>        [
        >>>           {'word': 'hello', 'postag': 'NNP'},
        >>>           {'word': 'world', 'postag': 'NN'},
        >>>           {'word': '@entity', 'postag': 'NN'}
        >>>        ],
        >>>        [
        >>>           {'word': 'hello', 'postag': 'NNP'},
        >>>           {'word': '@entity', 'postag': 'NN'}
        >>>        ]
        >>>    ]
        >>> })

    Output:
        >>> {
        >>>    '@entity': [
        >>>       np.matrix([
        >>>          [2, 1, 0],
        >>>          [2, 1, 0],
        >>>          [1, 1, 0]
        >>>       ]),
        >>>       np.matrix([
        >>>          [2, 1, 0],
        >>>          [1, 1, 0]
        >>>       ])
        >>>    ]
        >>> }
    """

    def __init__(self, vocab_words=None, vocab_postags=None, vocab_entities=None):
        """Initialize the tokenizer.

        Parameters
        ----------
        vocab_words : dict
            A mapping from word to word identifiers (word_id) used for the tokenization.
        vocab_postags : dict
            A mapping from POS-tag to POS-tag identifiers (postag_id) used for the tokenization.
        vocab_entities : dict
            A mapping from entity words to entity identifiers (entity_id) used for the tokenization.
        """
        self.vocab_words = vocab_words if vocab_words is not None else {'<PAD>': 0, '<UNK>': 1}
        self.vocab_postags = vocab_postags if vocab_postags is not None else {'<PAD>': 0, '<UNK>': 1}
        self.vocab_entities = vocab_entities if vocab_entities is not None else {'<PAD>': 0, '<UNK>': 1}

    def tokenize(self, word, postag=None):
        """Converts (word, postag) tuples to [word_id, postag_id, entity_id] tuples.

        Parameters
        ----------
        word : str
            The word to use.
        postag : str, optional
            The POS-tag to use (default: None).

        Returns
        -------
        tuple
            A [word_id, postag_id, entity_id] tuple.
        """
        word = unidecode.unidecode(word.lower()) if word not in self.vocab_entities.keys() else None
        entity = word if word in self.vocab_entities.keys() else None
        word_id = self.vocab_words.get(word, self.vocab_words.get('<UNK>'))
        postag_id = self.vocab_postags.get(postag, self.vocab_postags.get('<UNK>'))
        entity_id = self.vocab_entities.get(entity, self.vocab_entities.get('<PAD>'))
        return [word_id, postag_id, entity_id]

    def tokenize_window(self, window):
        """Tokenize a window.

        Parameters
        ----------
        window : list
            The window which is a list of items in which each item contains at least a 'word' field and a 'postag'
            field.

        Returns
        -------
        np.matrix
            A matrix with columns word_id_window, postag_id_window, entity_id_window in which:
             - word_id_window is a list of word_id elements which are the identifiers of the given words.
             - postag_id_window is a list of postag_id elements which are identifiers of the given POS-tags.
             - entity_id_window is a list of entity_id elements which are identifiers of the given entities.
            Furthermore, the rows of the matrix correspond to the different items in the given window.
        """

        return [self.tokenize(item['word'], item['pos']) for item in window]

    def tokenize_document(self, document):
        """Tokenize a document.

        Parameters
        ----------
        document : dict
            A mapping (dict) from entities to a list of windows in which the windows correspond to the given entities.

        Returns
        -------
        dict
            A mapping (dict) from entities to a list consisting of windows in which each window is a matrix in which the
            rows correspond to tokenized words (see the tokenize_window method for more details).
        """
        representation = {}
        for entity in document:
            representation[entity] = [self.tokenize_window(window) for window in document[entity]]
        return representation
