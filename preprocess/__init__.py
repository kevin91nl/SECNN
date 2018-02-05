import json
import re
from urllib.parse import urlencode

import requests
import unidecode


class StanfordCoreNLPClient:
    """A client for the Stanford CoreNLP server."""

    def __init__(self, corenlp_base_url):
        """Initialize the Stanford CoreNLP client.

        Parameters
        ----------
        corenlp_base_url : str
            The URL to the Stanford CoreNLP server.
        """
        self.session = requests.Session()
        self.corenlp_base_url = corenlp_base_url

    def __call__(self, text):
        """Query the Stanford CoreNLP server with text.

        Parameters
        ----------
        text : str
            Text to run the Stanford CoreNLP server for.

        Returns
        -------
        dict
            The JSON output of the Stanford CoreNLP server.
        """
        query = {
            "properties": {
                "annotators": "tokenize,ssplit,pos,ner,coref",
                "timeout": 20000,
            },
            "pipelineLanguage": "en"
        }
        url = '%s/?%s' % (self.corenlp_base_url, urlencode(query))
        response = self.session.post(url, text)
        return json.loads(response.text)


def corenlp_to_tokens(corenlp_data):
    """Converts the output of the Stanford CoreNLP client to a list of tokens.

    Parameters
    ----------
    corenlp_data : dict
        The output of the Stanford CoreNLP client.

    Returns
    -------
    list
        A list containing tokens (dicts). The tokens are used as input for other functions.
    """
    tokens = []
    for sentence_index, sentence in enumerate(corenlp_data['sentences']):
        for token_index, token in enumerate(sentence['tokens']):
            tokens += [{
                'word': token['originalText'],
                'ner': token['ner'],
                'pos': token['pos'],
                'sentence': sentence_index + 1,
                'index': token_index + 1
            }]
    return tokens


def get_entities(tokens):
    """Find all entities in the given tokens.

    Parameters
    ----------
    tokens : list
        Tokens found by the corenlp_to_tokens method.

    Returns
    -------
    list
        A list of lists in which each of the lists represents an entity. One entity consists of multiple tokens.
    """
    entities = []
    buffer = []
    current_ner = 'O'
    for token in list(tokens) + [{'ner': 'O'}]:
        ner = token['ner']
        if ner != current_ner:
            if len(buffer) > 0:
                entities.append(buffer)
            buffer = []
        if ner != 'O':
            buffer.append(token)
        current_ner = ner
    return entities


def tokens_to_text(tokens):
    """Convert a list of tokens to text.

    Parameters
    ----------
    tokens : list
        Tokens found by the corenlp_to_tokens method.

    Returns
    -------
    str
        String representation of the tokens.
    """
    return ' '.join([token['word'] for token in tokens])


def remove_empty_tokens(tokens):
    """Remove tokens with an empty word field.

    Parameters
    ----------
    tokens : list
        Tokens found by the corenlp_to_tokens method.

    Returns
    -------
    list
        List of tokens where empty tokens are removed.
    """
    return [token for token in tokens if len(token['word']) > 0]


def replace_entities(tokens, entities):
    """Replace entities by entity markers

    Parameters
    ----------
    tokens : list
        Tokens found by the corenlp_to_tokens method.

    entities : list
        Entities (or filtered entities) obtained from the get_entities method in which the first token of each entity
        has a 'label' attribute (for example obtained by the cluster_entities method).

    Returns
    -------
    list
        List of tokens in which the head token of an entity is replaced by an entity marker ("@entity...") and the
        remaining entity tokens are emptied. The entity marker head token also gets an additional field "entity" which
        is the text of entity mention.
    dict
        The entities enriched with a label attribute.
    """
    entities = entities  # type: list
    tokens = list(tokens)
    for entity_index, entity in enumerate(entities):
        label = entity[0]['label']
        start_index = entity[0]['index']
        end_index = entity[-1]['index']
        sentence = entity[0]['sentence']
        for token_index, token in enumerate(tokens):
            if start_index <= token['index'] <= end_index and token['sentence'] == sentence:
                if token['index'] == start_index:
                    tokens[token_index]['entity'] = tokens_to_text(entity)
                tokens[token_index]['word'] = label if token['index'] == start_index else ''
        for token_index, token in enumerate(entities[entity_index]):
            entities[entity_index][token_index]['label'] = label

    return tokens, entities


def normalize_entity(entity_text):
    """Normalize the string representation of entities such that it can be used for string comparisons.

    Parameters
    ----------
    entity_text : str
        Entity text to normalize.

    Returns
    -------
    str
        Normalized text (lower-cased, replaced special characters [á à ã ...] -> a and removed non-alphabetic
        characters).
    """
    entity = entity_text.lower()
    entity = unidecode.unidecode(entity)
    entity = re.sub('[^a-z]', '', entity)
    return entity


def align_entities(entity_labels, entities):
    """Align entities.

    Parameters
    ----------
    entity_labels : list
        String representations of the entities to align.
    entities : list
        List of entities obtained by the get_entities method.

    Returns
    -------
    list
        List of entities enriched in which the tokens of an entity are enriched with an 'aligned_with' field which maps
        to one of the labels specified in the entity_labels list whenever the normalized entity string representation
        matches the normalized label.
    """
    entities = entities  # type: list
    for index, nlp_entity in enumerate(entities):
        text = tokens_to_text(nlp_entity)
        for entity_label in entity_labels:
            if normalize_entity(text) == normalize_entity(entity_label):
                for subindex, token in enumerate(nlp_entity):
                    entities[index][subindex]['aligned_with'] = entity_label
    return entities


def get_entity_windows(entity, tokens, pre_window_size=15, post_window_size=15, pad_token='<PAD>',
                       replace_by_target=True):
    """Get a list of windows in which each window is a list of tokens centered around the token of the specified entity.

    Parameters
    ----------
    entity : dict
        An entity (one of the entities obtained by the get_entities method).
    tokens : list
        Tokens (obtained by the corenlp_to_tokens method.
    pre_window_size : int, optional
        Number of tokens before the entity token (default: 15).
    post_window_size : int, optional
        Number of tokens after the entity token (default: 15).
    pad_token : str, optional
        The PAD token (used for filling up empty space, default: '<PAD>').
    replace_by_target : bool, optional
        When true, then the word attribute of tokens equal to the word attribute of the specified entity is replaced by
        '@target' (default: True).

    Returns
    -------
    list
        A list of windows (in which a window is a list of tokens containing the entity token).
    """
    entity_label = entity[0]['label']
    windows = []
    for token_index, token in enumerate(tokens):
        if token['word'] == entity_label:
            pre_window = tokens[token_index - pre_window_size:token_index]
            post_window = tokens[token_index + 1:token_index + 1 + post_window_size]
            pre_window = max(0, pre_window_size - len(pre_window)) * [pad_token] + pre_window
            post_window = post_window + max(0, post_window_size - len(post_window)) * [pad_token]
            window = pre_window + [token] + post_window
            window = [{key: value for key, value in token.items()} for token in window]
            if replace_by_target:
                for window_index, window_token in enumerate(window):
                    window_token = window_token  # type: dict
                    if window_token['word'] == entity_label:
                        window[window_index]['word'] = '@target'
            windows.append(window)
    return windows


def cluster_entities(entities):
    """Cluster similar entities together such that they have the same 'label' attribute.

    Parameters
    ----------
    entities : list
        List of entities (obtained by the get_entities method).

    Returns
    -------
    list
        List of entities having a 'label' attribute in which entities which equivalent normalized word representations
        get the same label.
    """
    current_index = 1
    for index, entity in enumerate(entities):
        entity[0]['label'] = '@entity%d' % current_index
        current_index += 1
    for index1, entity1 in enumerate(entities):
        for index2, entity2 in enumerate(entities[index1 + 1:]):
            if normalize_entity(tokens_to_text(entity2)) == normalize_entity(tokens_to_text(entity1)):
                entity2[0]['label'] = entity1[0]['label']
    return entities
