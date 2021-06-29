import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import glob
import re
from tqdm import tqdm
import csv
import spacy
from spacy.tokens import DocBin



def convertDoccanoToSpacy(path_csv, LABEL):
    datasets = []
    csv_file = csv.reader(open(path_csv, "r"), delimiter="	")
    data = {}


    #for line in json_file:
    #    data = json.loads(line)
    #    data['entities'] = []
    #    id = data['id']
    #    for row in csv_file:
    #        if int(id) == int(row[0]):
                #data['labels2'] = list(removeduplicate(row[1]))
                #data['labels2'] = list(removeoverlapping(data['labels2']))
                #data['entities'].append(row[1])


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
    nlp = spacy.blank("en")  # load a new spacy model
    db = DocBin()  # create a DocBin object

    for text, annot in tqdm(TRAIN_DATA):  # data in previous format
        doc = nlp.make_doc(text)  # create doc object from text
        ents = []
        for start, end, label in annot["entities"]:  # add character indexes
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                pass
            else:
                ents.append(span)
        doc.ents = ents  # label the text with the ents
        db.add(doc)

    db.to_disk("./train.spacy")  # save the docbin object


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
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot.get('entities'))
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
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

        #svg = displacy.render(doc, style='dep')

        #output_path = os.path.join('./', 'NER.svg')
        #svg_file = open(output_path, "w", encoding="utf-8")
        #svg_file.write(svg)


def trainSpacyModel(path_csv, LABEL, dropout, nIter, model=None):
    dataset = convertDoccanoToSpacy(path_csv, LABEL)
    nlp = trainSpacy(dataset, dropout, nIter, model)

    return nlp


def evaluateSpacy(nlp, dataset):
    return evaluate(nlp, dataset)

def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())

# does not lowercase the text
def clean_text2(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt))
