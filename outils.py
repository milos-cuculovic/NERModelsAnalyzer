# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import csv

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import tensorflow
from spacy.scorer import Scorer
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.utils import plot_model
import keras
import pydot as pyd
from keras.utils.vis_utils import model_to_dot
import os
import random
import spacy
from spacy.util import minibatch, compounding
from spacy.gold import GoldParse
import warnings

keras.utils.vis_utils.pydot = pyd

from numpy.random import seed
from spacy.util import decaying

seed(1)
tensorflow.random.set_seed(2)


def get_dict_map(data, token_or_tag):
    tok2idx = {}
    idx2tok = {}

    if token_or_tag == 'token':
        vocab = list(set(data['Word'].to_list()))
    else:
        vocab = list(set(data['Tag'].to_list()))

    idx2tok = {idx: tok for idx, tok in enumerate(vocab)}
    tok2idx = {tok: idx for idx, tok in enumerate(vocab)}
    return tok2idx, idx2tok


def get_pad_train_test_val(data_group, data):
    # get max token and tag length
    n_token = len(list(set(data['Word'].to_list())))
    n_tag = len(list(set(data['Tag'].to_list())))

    # Pad tokens (X var)
    tokens = data_group['Word_idx'].tolist()
    maxlen = max([len(s) for s in tokens])
    pad_tokens = pad_sequences(tokens, maxlen=maxlen, dtype='int32', padding='post', value=n_token - 1)

    # Pad Tags (y var) and convert it into one hot encoding
    tags = data_group['Tag_idx'].tolist()
    pad_tags = pad_sequences(tags, maxlen=maxlen, dtype='int32', padding='post', value=tag2idx["O"])
    n_tags = len(tag2idx)
    pad_tags = [to_categorical(i, num_classes=n_tags) for i in pad_tags]

    # Split train, test and validation set
    tokens_, test_tokens, tags_, test_tags = train_test_split(pad_tokens, pad_tags, test_size=0.1, train_size=0.9,
                                                              random_state=2020)
    train_tokens, val_tokens, train_tags, val_tags = train_test_split(tokens_, tags_, test_size=0.25, train_size=0.75,
                                                                      random_state=2020)

    print(
        'train_tokens length:', len(train_tokens),
        '\ntrain_tokens length:', len(train_tokens),
        '\ntest_tokens length:', len(test_tokens),
        '\ntest_tags:', len(test_tags),
        '\nval_tokens:', len(val_tokens),
        '\nval_tags:', len(val_tags),
    )

    return train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags


def get_bilstm_lstm_model(input_dim, output_dim, n_tags, input_length):
    model = Sequential()

    # Add Embedding layer
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))

    # Add bidirectional LSTM
    model.add(Bidirectional(LSTM(units=output_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
                            merge_mode='concat'))

    # Add LSTM
    model.add(LSTM(units=output_dim, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))

    # Add timeDistributed Layer
    model.add(TimeDistributed(Dense(n_tags, activation="relu")))

    # Optimiser
    # adam = k.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def train_model(X, y, model):
    loss = list()
    for i in range(25):
        # fit model for one epoch on this sequence
        hist = model.fit(X, y, batch_size=1000, verbose=1, epochs=1, validation_split=0.2)
        loss.append(hist.history['loss'][0])
    return loss


def do_model():
    LABEL = ['LOCATION', 'CONTENT', 'TRIGGER', 'MODAL', 'ACTION']

    ROOT_DIR = os.path.dirname(os.path.abspath('data.json'))
    path_csv = os.path.join(ROOT_DIR, 'selena2.csv')

    dataset = convert_doccano_to_spacy(path_csv, LABEL)

    prdnlp = train_spacy(dataset, None, ROOT_DIR)

    # Save our trained Model
    modelfile = input("Enter your Model Name: ")
    prdnlp.to_disk(modelfile)

    # Test your text
    test_text = input("Enter your testing text: ")
    doc = prdnlp(test_text)
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)

    evaluate(prdnlp, dataset)

    exit()
    data.head()

    token2idx, idx2token = get_dict_map(data, 'token')
    tag2idx, idx2tag = get_dict_map(data, 'tag')

    data['Word_idx'] = data['Word'].map(token2idx)
    data['Tag_idx'] = data['Tag'].map(tag2idx)
    data.head()

    # Fill na
    data_fillna = data.fillna(method='ffill', axis=0)  # Group by and collect columns
    data_group = data_fillna.groupby(
        ['Sentence #'], as_index=False
    )['Word', 'POS', 'Tag', 'Word_idx', 'Tag_idx'].agg(lambda x: list(x))  # Visualise data
    data_group.head()


    train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags = get_pad_train_test_val(data_group, data)

    input_dim = len(list(set(data['Word'].to_list()))) + 1
    output_dim = 64
    input_length = max([len(s) for s in data_group['Word_idx'].tolist()])
    n_tags = len(tag2idx)
    print('input_dim: ', input_dim, '\noutput_dim: ', output_dim, '\ninput_length: ', input_length, '\nn_tags: ',
          n_tags)

    results = pd.DataFrame()
    model_bilstm_lstm = (input_dim, output_dim, n_tags)
    plot_model(model_bilstm_lstm)
    results['with_add_lstm'] = train_model(train_tokens, np.array(train_tags), model_bilstm_lstm)


def convert_doccano_to_spacy(path_csv, LABEL):
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
    #            data['entities'].append(row[1])


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

            if str(labels_formated2[2]) != 'CONTENT':
                labels_formated.append([int(labels_formated2[0]), int(labels_formated2[1]), str(labels_formated2[2])])

        data['text'] = row[1]

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


def train_spacy(TRAIN_DATA, model=None, output_dir=None, n_iter=500):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    dropout = decaying(0.6, 0.2, 1e-4)

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            optimizer = nlp.begin_training()

        # Add special case rule
        infixes = (":","“",",", '“', "/", ";", "-", ".", '”') + nlp.Defaults.infixes
        infix_regex = spacy.util.compile_infix_regex(infixes)
        nlp.tokenizer.infix_finditer = infix_regex.finditer

        # Check new tokenization
        # print([w.text for w in nlp('25) Figure 8:“pines are results of two-step adsorption model” -> What method/software was used for the curve fitting?')])
        # exit()

        #for train in TRAIN_DATA:
        #    print(train[0])
        #    print([w.text for w in nlp(train[0])])
        #    print("\n")

        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}

            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)

                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                    sgd=optimizer,
                )
            print(itn)
            print("Losses", losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

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

def testspacymodel(test_text, output_dir):
    # Test the saved model
    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir)
    doc2 = nlp2(test_text)
    for ent in doc2.ents:
        print(ent.label_, ent.text)

def evaluate(ner_model, examples):

    scorer = Scorer()
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot.get('entities'))
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return scorer.scores


if __name__ == '__main__':
    do_model()
