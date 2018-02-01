def corenlp_to_tokens(corenlp_data):
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
    entities = []
    buffer = []
    current_ner = 'O'
    for token in tokens + [{'ner': 'O'}]:
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

    print(entities)

    return tokens


def remove_empty_tokens(tokens):
    return [token for token in tokens if len(token['word']) > 0]
