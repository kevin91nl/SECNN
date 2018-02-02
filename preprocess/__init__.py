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


def replace_entities(results, entities, tokens):
    current_id = 1
    for cluster in results['corefs']:
        for item in results['corefs'][cluster]:
            label = '@%s%d' % ('entity', current_id)
            for token_index, token in enumerate(tokens):
                if item['startIndex'] <= token['index'] < item['endIndex'] and token['sentence'] == item['sentNum']:
                    tokens[token_index]['word'] = label if token['index'] == item['startIndex'] else ''
        current_id += 1
    return tokens


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
