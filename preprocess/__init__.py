import json
from urllib.parse import urlencode

import requests


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


def replace_entities(tokens, allowed_entity_types=None, entity_id=1):
    """Replace entities by entity markers

    Parameters
    ----------
    tokens : list
        Tokens found by the corenlp_to_tokens method.

    allowed_entity_types : list, optional
        A list of all NER types that are used for the replacement. All other NER types are ignored. The default is
        ['LOCATION', 'ORGANIZATION', 'PERSON', 'MISC'].

    entity_id : int, optional
        The entity index to start with (default: 1).

    Returns
    -------
    list
        List of tokens in which the head token of an entity is replaced by an entity marker ("@entity...") and the
        remaining entity tokens are emptied. The entity marker head token also gets an additional field "entity" which
        is the text of entity mention.
    int
        The maximum found entity index.
    """
    allowed_entity_types = allowed_entity_types if allowed_entity_types is not None else ['LOCATION', 'ORGANIZATION',
                                                                                          'PERSON', 'MISC']
    tokens = list(tokens)
    entities = get_entities(tokens)
    entities = [entity for entity in entities if len(entity) > 0 and entity[0]['ner'] in allowed_entity_types]
    max_entity_id = entity_id
    for entity in entities:
        max_entity_id = entity_id
        label = '@entity%d' % entity_id
        start_index = entity[0]['index']
        end_index = entity[-1]['index']
        sentence = entity[0]['sentence']
        for token_index, token in enumerate(tokens):
            if start_index <= token['index'] <= end_index and token['sentence'] == sentence:
                if token['index'] == start_index:
                    tokens[token_index]['entity'] = tokens_to_text(entity)
                tokens[token_index]['word'] = label if token['index'] == start_index else ''
        entity_id += 1

    return tokens, max_entity_id


def replace_corefs(corenlp_data, tokens, entity_id):
    """Replace entities by entity markers

    Parameters
    ----------
    corenlp_data : dict
        The output of the Stanford CoreNLP client.

    tokens : list
        Tokens found by the corenlp_to_tokens method..

    entity_id : int, optional
        The entity index to start with (default: 1).

    Returns
    -------
    list
        List of tokens in which the head token of an entity is replaced by an entity marker ("@entity...") and the
        remaining entity tokens are emptied. The entity marker head token also gets an additional field "entity" which
        is the text of entity mention. Entities from the same coreference resolution cluster are assigned the same
        label.
    int
        The maximum found entity index.
    """
    max_entity_id = entity_id
    tokens = list(tokens)
    for cluster in corenlp_data['corefs']:
        max_entity_id = entity_id
        for entity in corenlp_data['corefs'][cluster]:
            start_index = entity['startIndex']
            end_index = entity['endIndex']
            sentence = entity['sentNum']
            label = '@%s%d' % ('entity', entity_id)
            for token_index, token in enumerate(tokens):
                if start_index <= token['index'] < end_index and token['sentence'] == sentence:
                    if token['index'] == start_index:
                        tokens[token_index]['entity'] = tokens_to_text(entity)
                    tokens[token_index]['word'] = label if token['index'] == entity['startIndex'] else ''
        entity_id += 1

    return tokens, max_entity_id
