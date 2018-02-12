import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, report

from utils.neural_network import DummyLoss


class SECNN(Chain):

    def __init__(self, config_word=None, config_postag=None, config_entity=None, config_rnn=None, config_affine=None):
        config_word = config_word if config_word is not None else {}
        config_postag = config_postag if config_postag is not None else {}
        config_entity = config_entity if config_entity is not None else {}
        config_rnn = config_rnn if config_rnn is not None else {}
        config_affine = config_affine if config_affine is not None else {}
        super(SECNN, self).__init__()
        with self.init_scope():
            self.embed_word = L.EmbedID(**config_word)
            self.embed_postag = L.EmbedID(**config_postag)
            self.embed_entity = L.EmbedID(**config_entity)
            self.rnn = L.LSTM(**config_rnn)
            self.affine = L.Linear(**config_affine)

    def __call__(self, minibatch, *args, **kwargs):
        y_batched = []
        for document in minibatch:
            y = {}
            for entity in document:
                y_windows = []
                for window in document[entity]:
                    word_ids = np.array([item[0] for item in window])
                    postag_ids = np.array([item[1] for item in window])
                    entity_ids = np.array([item[2] for item in window])
                    x_word = self.embed_word(word_ids)
                    x_postag = self.embed_postag(postag_ids)
                    x_entity = self.embed_entity(entity_ids)
                    x_seq = F.concat([x_word, x_postag, x_entity], axis=-1)
                    self.rnn.reset_state()
                    h_rnn = self.rnn(x_seq)
                    y_windows.append(h_rnn)
                y_sentences = F.concat(y_windows, axis=0)
                y_entity = F.mean(y_sentences, axis=0)
                y_entity = F.expand_dims(y_entity, 0)
                y[entity] = self.affine(y_entity)
            y_batched.append(y)
        return y_batched


class SECNNLossWrapper(Chain):

    def __init__(self, model, **links):
        super().__init__(**links)
        self.model = model

    def __call__(self, minibatch, *args, **kwargs):
        docs = [item['document'] for item in minibatch]
        targets = [item['targets'] for item in minibatch]
        entity_scores = self.model.__call__(docs, *args, **kwargs)

        losses = []
        for doc_index, doc in enumerate(entity_scores):
            for entity in doc:
                y_out = doc[entity]
                target = targets[doc_index][entity]

                losses.append((target - y_out) ** 2)

        if len(losses) == 0:
            return DummyLoss()

        loss = F.mean(F.concat(losses))

        report({
            'loss': loss
        })

        return loss
