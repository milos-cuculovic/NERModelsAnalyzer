import csv
import warnings
import spacy
import os

from numpy import random
from spacy.util import minibatch, compounding
from spacy.scorer import Scorer
from spacy import displacy
from spacy.training import Example


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


def trainSpacy(TRAIN_DATA, dropout, nIter, model=None):

    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")
        print("Created blank 'en' model")

    if "ner" not in nlp.pipe_names:
        nlp.add_pipe("ner", last=True)
    else:
        nlp = nlp.get_pipe("ner")

    examples = []
    # add labels
    for text, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            examples.append(Example.from_dict(nlp.make_doc(text), annotations))

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "tagger", "parser", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    #only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            optimizer = nlp.initialize(lambda: examples)

        # Add special case rule
        infixes = list(nlp.Defaults.infixes)
        #infixes.extend((":","“",",", '“', "/", ";", "-", ".", '”'))
        infixes.extend((":", "“", ",", '“', "/", ";", ".", '”'))
        infix_regex = spacy.util.compile_infix_regex(infixes)
        nlp.tokenizer.infix_finditer = infix_regex.finditer

        for itn in range(nIter):
            random.shuffle(examples)
            losses = {}

            # batch up the examples using spaCy's minibatch
            batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                #texts, annotations = zip(*batch)

                nlp.update(
                    batch,
                    drop=dropout,  # dropout - make it harder to memorise data
                    losses=losses,
                    sgd=optimizer,
                )

            print(itn)
            print("Losses", losses)

    return nlp


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


def evaluate(ner_model, examples):
    scorer = Scorer()
    for input_, annot in examples:
        doc = ner_model.make_doc(input_)
        example = Example.from_dict(doc, {"entities": annot.get('entities')})
        pred_value = ner_model(input_)
        scorer.score(pred_value, example)

    return scorer.scores


def testSpacyModel(model_name, number_of_testing_examples):
    nlp = spacy.load(model_name)

    print(nlp.pipe_names)

    for x in range(int(number_of_testing_examples)):
        test_text = input("Enter your testing text: ")
        doc = nlp(str(test_text))
        for ent in doc.ents:
            #print(ent.text, ent.start_char, ent.end_char, ent.label_)
            print(ent.text, ent.label_)

        svg = displacy.render(doc, style='dep')

        output_path = os.path.join('./', 'NER.svg')
        svg_file = open(output_path, "w", encoding="utf-8")
        svg_file.write(svg)


def trainSpacyModel(path_csv, LABEL, dropout, nIter, model=None):
    dataset = convertDoccanoToSpacy(path_csv, LABEL)
    nlp = trainSpacy(dataset, dropout, nIter, model)

    return nlp


def evaluateSpacy(nlp, dataset):
    return evaluate(nlp, dataset)