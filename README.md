# Salient Entity Classifier Neural Network (SECNN)

This repository is the implementation of the Salient Entity Classifier Neural Network (SECNN) model. The script is written using Python 3.6.1.

## Model input

The input for the model consists of a collection of JSON files. Each of the JSON files should describe one document. The input path argument should point to the directory containing the input files.

### JSON structure

The minimum requirements for the JSON files are that they have a `text` field. In that case, the JSON file looks like the following:

```
{
    "text": "This is a document. The document contains sentences."
}
```

When the JSON file is used for training and testing, information about entity salience should be given. The structure should contain a `salient_entities` field which has a list as its value containing mentions of the entities that are salient. The JSON files used for training and testing are similar to the following example:

```
{
    "text": "This is a document about cats and dogs. Dogs are animals and cats too.",
    "salient_entities": ["dogs", "cats"]
}
```

More information is better. An optional field is the `title` field which should contain the title of the document when this information is available.

This is the basic structure of the input files. There are required preprocessing steps which add extra information to the JSON files but do not modify the required fields.

## Preprocessing

The preprocessing of the input data consists of several steps. First, NLP annotations are added to the data. Then, entity alignment is applied whenever salient entities are specified. With the alignment, each of the specified salient entities are aligned with the entities found by the NLP pipeline. First, NLP annotations are added. This is done using the Stanford CoreNLP pipeline. Here, version 3.8.0 is used. The `preprocessing.py` script adds first an `nlp_annotations` field to the given JSON files. Before this script is called, make sure that the Stanford CoreNLP server runs on port `http://localhost:9000` (or on a different URL/port and by changing the `corenlp_url` argument of the preprocessing script). When the `nlp_annotations` are added and when salient entities are specified in the `salient_entities` field, then entity alignments are made and stored in the `aligned_entities` field, which is a mapping from salient entities to the entities found by the NLP pipeline.
