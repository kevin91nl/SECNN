import argparse
import os

from preprocess import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the SECNN model on the given preprocessed input files.')
    parser.add_argument('input',
                        help='Path to the input files (folder containing preprocessed JSON files).')
    args = parser.parse_args()

    files = os.listdir(args.input)
    for file in files[:1]:
        path = os.path.join(args.input, file)
        with open(path, 'r') as file_handle:
            data = json.load(file_handle)

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
