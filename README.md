# Salient Entity Classifier Neural Network

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

When the JSON file is used for training and testing, information about entity salience should be given. The structure should contain a `salient_entities` field which has a list as its value containing mentions of the entities that are salient. Besides that, it should contain a `nonsalient_entities` field which has a list as its value containing the mentions of the entities that are non-salient. The JSON files used for training and testing are similar to the following example:

```
{
    "text": "This is a document about cats and dogs. Dogs are animals and cats too.",
    "salient_entities": ["dogs", "cats"],
    "nonsalient_entities": ["document"]
}
```

More information is better. An optional field is the `title` field which should contain the title of the document when this information is available.

This is the basic structure of the input files. There are required preprocessing steps which add additional information to the JSON files but it will not modify the existing fields.

## Preprocessing

### NLP pipeline

One required preprocessing step is annotating the data with the annotations found by the Stanford CorenLP package. This is done by executing the `preprocess_nlp.py` script. As input, it needs the path to the folder in which the JSON files are stored. It will not modify the existing fields, but only add the annotations in the `nlp_data` field. Make sure that the Stanford CoreNLP server is running. If the server is running on a different URL than `http://localhost:9000`, make sure to adjust the `corenlp_url` argument of the script accordingly.

## Training

After the preprocessing is done, the JSON files are used as input for the train script. The train script is called as follows:

```
python train.py
```

By default, the log file and snapshots are created in the `result` folder. It is possible to resume from a given snapshot by specifying the `result` argument. The documentation for the arguments are found by calling the help function:

```
python train.py --help
```