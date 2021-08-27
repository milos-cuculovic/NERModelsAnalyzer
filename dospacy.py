import warnings

import numpy as np
import spacy
import os
import json
import matplotlib.pyplot as plt

from numpy import random
from spacy.util import minibatch, compounding
from spacy.scorer import Scorer
from spacy import displacy
from spacy.training import Example
from thinc.api import SGD, RAdam, Adam


def convertDoccanoToSpacy(path_csv, LABEL):
    datasets = []
    csv_file = csv.reader(open(path_csv, "r"), delimiter="	")
    data = {}

    for row in csv_file:
        labels_formated = []
        row[2] = row[2].replace('{', '')
        row[2] = row[2].replace('}', '')
        row[2] = row[2].replace('"', '')
        row[2] = row[2].replace("'", '')
        row[2] = row[2].replace(', ', '-')
        labels = row[2].split(',')

        for label in labels:
            labels_formated2 = label.replace('-', ',').split(',')

            #if str(labels_formated2[2]) != 'CONTENT':
            labels_formated.append([int(labels_formated2[0]), int(labels_formated2[1]), str(labels_formated2[2])])

        data['text'] = row[1].replace(' ',' ')

        data['labels'] = list(removeDuplicate(labels_formated))
        data['labels'] = list(removeOverlapping(data['labels']))
        data['labels'] = list(removeBlankSpaces(data['labels'], data['text']))

        tmp_ents = []

        for e in data["labels"]:
            if e[2] in LABEL:
                tmp_ent = (e[0], e[1], e[2])
                tmp_ents.append(tmp_ent)
            data["entities"] = tmp_ents

        if len(data["text"]) > 5:
            dataset = (data["text"], {"entities": data["entities"]})
            datasets.append(dataset)

    return datasets


def convertJsonToSpacy(path_train_data, LABEL):

    labeled_data = []
    with open(path_train_data, "r") as read_file:
        for line in read_file:
            data = json.loads(line)
            labeled_data.append(data)

    TRAINING_DATA = []

    for entry in labeled_data:
        entities = []
        entry['labels'] = list(removeDuplicate(entry['labels']))
        entry['labels'] = list(removeOverlapping(entry['labels']))
        entry['labels'] = list(removeBlankSpaces(entry['labels'], entry['text']))

        for e in entry['labels']:
            entities.append((e[0], e[1], e[2]))
        spacy_entry = (entry['text'], {"entities": entities})
        TRAINING_DATA.append(spacy_entry)

    return TRAINING_DATA

def trainSpacy(TRAIN_DATA, VALID_DATA, dropout, nIter, modelFile, model=None):

    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")
        print("Created blank 'en' model")

    #if "transformer" not in nlp.pipe_names:
    #    nlp.add_pipe("transformer", last=True)

    if "ner" not in nlp.pipe_names:
        nlp.add_pipe("ner", last=True)
    else:
        nlp = nlp.get_pipe("ner")

    # Add special case rule
    infixes = list(nlp.Defaults.infixes)
    infixes.extend((":", "“", ",", '“', "/", ";", "\.", '”'))
    infix_regex = spacy.util.compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_regex.finditer

    examples_train = []
    # add labels
    for text, annotations in TRAIN_DATA:
        examples_train.append(Example.from_dict(nlp.make_doc(text), annotations))

    examples_valid = []
    # add labels
    for text, annotations in VALID_DATA:
        examples_valid.append(Example.from_dict(nlp.make_doc(text), annotations))

    # get names of other pipes to disable them during training
    #pipe_exceptions = ["ner", "tagger", "parser", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    #only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            #optimizer_default = nlp.initialize()
            optimizer_adam = Adam(
                learn_rate=0.001,
                beta1=0.9,
                beta2=0.999,
                eps=1e-08,
                L2=1e-6,
                grad_clip=1.0,
                use_averages=True,
                L2_is_weight_decay=True
            )

            optimizer_sdg = SGD(
                learn_rate=0.001,
                L2=1e-6,
                grad_clip=1.0
            )

            optimizer_radam = RAdam(
                learn_rate=0.001,
                beta1=0.9,
                beta2=0.999,
                eps=1e-08,
                L2_is_weight_decay=True,
                grad_clip=1.0,
                use_averages=True,
            )

            #optimizer_lambda = nlp.initialize(lambda: examples)

            nlp.initialize()
            losses_train_history = []
            losses_valid_history = []

        for itn in range(nIter):
            print("Iteration " + str(itn))
            random.shuffle(examples_train)
            losses_train = {}

            batches = minibatch(examples_train, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                try:
                    nlp.update(
                        batch,
                        drop=dropout,  # dropout - make it harder to memorise data
                        losses=losses_train,
                        sgd=optimizer_radam,
                    )
                except Exception as error:
                    #print(error)
                    continue

            print("Losses train", losses_train)
            losses_train_history.append(losses_train['ner'])

            losses_valid = {}
            batches2 = minibatch(examples_valid, size=compounding(4.0, 32.0, 1.001))
            for batch2 in batches2:
                try:
                    nlp.update(
                        batch2,
                        sgd=None,
                        losses=losses_valid
                    )
                except Exception as error:
                    #print(error)
                    continue

            nlp.use_params(optimizer_radam.averages)

            print("Losses valid", losses_valid)
            losses_valid_history.append(losses_valid['ner'])

        plt.plot(np.array(list(losses_train_history)).astype(float))
        plt.plot(np.array(list(losses_valid_history)).astype(float))

    return nlp, plt


def removeDuplicate(it):
    seen = []
    for x in it:
        if x not in seen:
            yield x
            seen.append(x)
    return seen


def removeOverlapping(it):
    seen = []
    it2 = it.copy()
    for x in it:
        overlap = False
        it2.remove(x)
        if it2:
            for y in it2:
                xs = set(range(x[0], x[1]))
                z = xs.intersection(range(y[0], y[1]))
                if len(z) > 0:
                    overlap = True

        if overlap is False:
            seen.append(x)

    return seen

def removeBlankSpaces(it, text):
    seen = []
    for x in it:
        word = text[x[0]:x[1]]
        if word.startswith(" "):
            x[0] = int(x[0])+1
        if word.endswith(" "):
            x[1] = int(x[1])-1
        seen.append(x)
    return seen

def evaluate(nlp, TEST_DATA):
    examples = []
    for text, annotations in TEST_DATA:
        examples.append(Example.from_dict(nlp.make_doc(text), annotations))

    #return scorer.score_tokenization(examples)
    return nlp.evaluate(examples)


def testSpacyModel(model_name, number_of_testing_examples):
    nlp = spacy.load(model_name)

    print(nlp.pipe_names)

    for x in range(int(number_of_testing_examples)):
        test_text = input("Enter your testing text: ")
        doc = nlp(str(test_text))
        for ent in doc.ents:
            print(ent.text, ent.label_)


def trainSpacyModel(path_train_data, path_valid_data, LABEL, dropout, nIter, modelFile):
    TRAIN_DATA = convertJsonToSpacy(path_train_data, LABEL)
    VALID_DATA = convertJsonToSpacy(path_valid_data, LABEL)

    nlp, plt = trainSpacy(TRAIN_DATA, VALID_DATA, dropout, nIter, modelFile)

    return nlp, plt


def evaluateSpacy(model_name, path_test_data, LABEL):
    nlp = spacy.load(model_name)
    TESTING_DATA = convertJsonToSpacy(path_test_data, LABEL)
    return evaluate(nlp, TESTING_DATA)
