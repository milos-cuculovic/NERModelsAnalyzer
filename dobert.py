"""
Created on Fri Jun 25 17:49:48 2021
@author: chams
refactored by Milos - 07.09.2022
"""

import os
import json
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from pytorch_transformers import AdamW, BertTokenizer, WarmupLinearSchedule
from transformers import BertConfig, BertTokenizer, BertForTokenClassification
from transformers import RobertaConfig, RobertaTokenizer, RobertaForTokenClassification
from transformers import pipeline, AutoModelForTokenClassification
import torch
from tqdm import tqdm, trange
from seqeval.metrics import classification_report
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, RandomSampler
from nltk import word_tokenize
import torch.nn as nn
import cf_matrix
from bertconf import removEsc, sentenceMean, json_conll, trigConll, crossval, changeToOther
import shutil
from grid_search_results_print import print_3D_graph
import itertools

device = 'cpu'
seed = 42
max_seq_length = 128
do_lower_case = "store_false"


labelremove=["CONTENT"]
lab_list=["O", "B-LOCATION", "I-LOCATION", "B-TRIGGER", "I-TRIGGER",
                "B-MODAL", "I-MODAL", "B-ACTION", "I-ACTION", "B-CONTENT", "I-CONTENT", "[CLS]", "[SEP]"]


class NertrainBERT(BertForTokenClassification):
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                valid_ids=None, attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            # attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class NertrainRoBERTa(RobertaForTokenClassification):
    def forward(self, input_ids, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None):
        sequence_output = self.roberta(input_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class BertNer:
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,
                                   device='cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(batch_size):
            self.ber
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)
        return logits


class Ner:
    def __init__(self,model_dir: str):
        self.model , self.tokenizer, self.model_config = self.load_model(model_dir)
        self.label_map = self.model_config["label_map"]
        self.max_seq_length = self.model_config["max_seq_length"]
        self.label_map = {int(k):v for k,v in self.label_map.items()}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_dir: str, model_config: str = "model_config.json"):
        model_config = os.path.join(model_dir,model_config)
        model_config = json.load(open(model_config))
        model = BertNer.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=model_config["do_lower"])
        return model, tokenizer, model_config

    def tokenize(self, text: str):
        """ tokenize input"""
        words = word_tokenize(text)
        tokens = []
        valid_positions = []
        for i,word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions

    def preprocess(self, text: str):
        """ preprocess """
        tokens, valid_positions = self.tokenize(text)
        ## insert "[CLS]"
        tokens.insert(0,"[CLS]")
        valid_positions.insert(0,1)
        ## insert "[SEP]"
        tokens.append("[SEP]")
        valid_positions.append(1)
        segment_ids = []
        for i in range(len(tokens)):
            segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            valid_positions.append(0)
        return input_ids,input_mask,segment_ids,valid_positions

    def predict(self, text: str):
        input_ids,input_mask,segment_ids,valid_ids = self.preprocess(text)
        input_ids = torch.tensor([input_ids],dtype=torch.long,device=self.device)
        input_mask = torch.tensor([input_mask],dtype=torch.long,device=self.device)
        segment_ids = torch.tensor([segment_ids],dtype=torch.long,device=self.device)
        valid_ids = torch.tensor([valid_ids],dtype=torch.long,device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask,valid_ids)
        logits = F.softmax(logits,dim=2)
        logits_label = torch.argmax(logits,dim=2)
        logits_label = logits_label.detach().cpu().numpy().tolist()[0]

        logits_confidence = [values[label].item() for values,label in zip(logits[0],logits_label)]

        logits = []
        pos = 0
        for index,mask in enumerate(valid_ids[0]):
            if index == 0:
                continue
            if mask == 1:
                logits.append((logits_label[index-pos],logits_confidence[index-pos]))
            else:
                pos += 1
        logits.pop()

        labels = [(self.label_map[label],confidence) for label,confidence in logits]
        words = word_tokenize(text)
        assert len(labels) == len(words)
        output = [{"word":word,"tag":label,"confidence":confidence} for word,(label,confidence) in zip(words,labels)]
        return output


def prepare_datasets():
    # # # INITIAL
    # removEsc(os.path.abspath("train.json"))

    # # STEP ONE cross validation
    # crossval(os.path.abspath("train.json"), os.path.abspath(""))

    # # STEP TWO remove sentence without action and location
    # sentenceMean(os.path.abspath("train1.json"))

    # # STEP THREE convert json to conll
    # json_conll(os.path.abspath("train1.json"), os.path.abspath(""), 'train_final_1.txt')
    # json_conll(os.path.abspath("valid1.json"), os.path.abspath(""), 'valid_final_1.txt')

    # # # STEP FOUR REPLACE TRIGGER
    # trigConll(os.path.abspath("train_final_1.txt"), trigger)
    # trigConll(os.path.abspath("valid_final_1.txt"), trigger)

    shutil.copyfile(r'train.txt', r'train_temp.txt')
    shutil.copyfile(r'valid.txt', r'valid_temp.txt')
    shutil.copyfile(r'test.txt', r'test_temp.txt')

    for i in labelremove:
        changeToOther(i, "train_temp.txt")
        changeToOther(i, "valid_temp.txt")
        changeToOther(i, "test_temp.txt")
        lab_list.remove("I-" + i)
        lab_list.remove("B-" + i)


def train_bert_based_models(modelType, modelPath, useCuda, nIterations):
    prepare_datasets()

    learningRate        = 0.0001
    weightDecay         = 0.1
    warmupProportion    = 0.1
    trainBatchSize      = 8

    train(modelType, modelPath, useCuda, nIterations,
          learningRate, weightDecay, warmupProportion, trainBatchSize, index=0)

def grid_bert_based_models(modelType, modelPath, useCuda, nIterations):
    prepare_datasets()

    learningRate        = [1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3]
    weightDecay         = [0.1, 0.01, 0.001, 0.0001]
    warmupProportion    = [0.1]
    trainBatchSize	    = [128, 32, 24, 18, 12, 8]

    #learningRate       = [0.01, 0.001, 0.0001, 0.00001]
    #weightDecay        = [0.1, 0.01, 0.001, 0.0001]
    #warmupProportion   = [0.1]
    #trainBatchSize     = [128, 64, 32, 16]

    hyperparam          = [weightDecay, learningRate, warmupProportion, trainBatchSize]
    k                   = 0

    combinations = list(itertools.product(*hyperparam))
    shutil.copyfile(r'train.txt', r'train_temp.txt')
    shutil.copyfile(r'valid.txt', r'valid_temp.txt')
    shutil.copyfile(r'test.txt', r'test_temp.txt')

    for i in labelremove:
        changeToOther(i, "train_temp.txt")
        changeToOther(i, "valid_temp.txt")
        changeToOther(i, "test_temp.txt")
        lab_list.remove("I-"+i)
        lab_list.remove("B-"+i)

    for combination in combinations:
        k += 1
        weightDecay = combination[0]
        learningRate = combination[1]
        warmupProportion = combination[2]
        trainBatchSize = combination[3]

        train(modelType, modelPath, useCuda, nIterations,
              learningRate, weightDecay, warmupProportion, trainBatchSize, index=k)

    compare_grid_search_results(modelType, combinations, modelPath)



def test_bert_based_models(modelType, modelPath):
    prepare_datasets()
    test(modelType, modelPath)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask


def readfile(filename):
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))
    return data


class NerProcessor():
    """Processor for the CoNLL-2003 data set."""

    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        return readfile(input_file)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_temp.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid_temp.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid_temp.txt")), "test")

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def compare_grid_search_results(modelType, combinations, modelPath):
    results = {}
    gridSearch = {}

    for i in range(0, len(combinations)):
        with open(modelPath + "/" + str(i+1) + "/eval_results.txt") as file:
            for line in file:
                line[0].split()
                listword = line.split()
                if len(listword) > 0:
                    if listword[0] == "LOCATION":
                        precision_loc, recall_loc, f1score_loc \
                            = get_best_grid_scores(precision_loc, recall_loc, f1score_loc, listword, i + 1)
                        results['LOCATION'] = [precision_loc, recall_loc, f1score_loc]

                    if listword[0] == "weighted":
                        precision_wght, recall_wght, f1score_wght \
                            = get_best_grid_scores(precision_wght, recall_wght, f1score_wght, listword[1:], i + 1)
                        results['weighted'] = [precision_wght, recall_wght, f1score_wght]
                        weightDecay = combinations[i][0]
                        learningRate = combinations[i][1]
                        trainBatchSize = combinations[i][3]
                        gridSearch[i] = [weightDecay, learningRate, trainBatchSize, listword[4]]

    for result in results:
        print(result)
        print("   precision n " + str(results[result][0][0]) + " - " + str(results[result][0][1]))
        print("   recall n " + str(results[result][1][0]) + " - " + str(results[result][1][1]))
        print("   f1score n " + str(results[result][2][0]) + " - " + str(results[result][2][1]))

    print_3D_graph(gridSearch, modelPath, modelType)


def get_best_grid_scores(precision, recall, f1score, listword, i):
    if precision[1] < float(listword[1]):
        precision[1] = float(listword[1])
        precision[0] = i
    if recall[1] < float(listword[2]):
        recall[1] = float(listword[2])
        recall[0] = i
    if f1score[1] < float(listword[3]):
        f1score[1] = float(listword[3])
        f1score[0] = i

    return precision, recall, f1score
                  
def convert_examples_to_features(examples, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(lab_list, 1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []

        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]

            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])

        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])

        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)

        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask))
    return features


def pip_aggregation(model_name, new_model_name):
    model_path = os.path.dirname(os.path.abspath(__file__)) + '/trained_models/' + model_name
    new_model_path = os.path.dirname(os.path.abspath(__file__)) + '/trained_models/' + new_model_name
    mode = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenize = BertTokenizer.from_pretrained(model_path)
    nlp_ner = pipeline(
        "ner",
        aggregation_strategy="SIMPLE",
        model=mode,
        tokenizer=tokenize
    )
    nlp_ner.save_pretrained(new_model_path)


def prediction_bert_based_models(text, model_name):
    text = text.replace("\n", " ")
    model_path = os.path.dirname(os.path.abspath(__file__)) + '/trained_models/' + model_name
    mode = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenize = BertTokenizer.from_pretrained(model_path)

    nlp_ner = pipeline(
        "ner",
        model=mode,
        tokenizer=tokenize
    )

    result = nlp_ner(text)
    return result
    listWords = {}
    for label in label_list:
        if label not in ("O", "[CLS]", "[SEP]"):
            listWords[label[2:]] = []

    index_manual = 0

    for dic in result:
        label = dic['entity']
        label_index = label.find("_") + 1
        number = label[label_index:]
        pos = int(number) - 1
        label_name = label_list[pos]
        word = dic['word']

        if "##" in word or word in punctuation:
            continue
        else:
            index_manual += 1

        if label_name not in ("O", "[CLS]", "[SEP]"):
            listWords[label_name[2:]].append(index_manual)

    return listWords


def train(modelType, modelPath, useCuda, nIterations,
          learningRate, weightDecay, warmupProportion, trainBatchSize, index):
    maxGradNorm = 1.0
    adamEpsilon = 1e-8
    b1 = 0.9
    b2 = 0.999

    num_labels = len(lab_list) + 1
    processors = {"ner": NerProcessor}
    task_name = "ner".lower()
    processor = processors[task_name]()

    if modelType in ["bert-base-cased", "allenai/scibert_scivocab_cased", "distilbert-base-cased"]:
        tokenizer = BertTokenizer.from_pretrained(modelType, do_lower_case=do_lower_case)
        config = BertConfig.from_pretrained(modelType, num_labels=num_labels, finetuning_task=task_name)
        model = NertrainBERT.from_pretrained(modelType, from_tf=False, config=config)
    elif modelType == "roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained(modelType, do_lower_case=do_lower_case)
        config = RobertaConfig.from_pretrained(modelType, num_labels=num_labels, finetuning_task=task_name)
        model = NertrainRoBERTa.from_pretrained(modelType, from_tf=False, config=config)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if index != 0:
        modelPath=modelPath+str(index)
    if os.path.exists(modelPath) and os.listdir(modelPath):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(modelPath))
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)

    trainExamples = None
    numTrainOptimizationSteps = 0.0

    train_examples = processor.get_train_examples("")
    numTrainOptimizationSteps = int((len(train_examples) / trainBatchSize) * nIterations)

    if (useCuda):
        model.cuda()

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weightDecay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmupSteps = int(warmupProportion * numTrainOptimizationSteps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learningRate, betas=(b1, b2), eps=adamEpsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmupSteps, t_total=numTrainOptimizationSteps)


    label_map = {i: label for i, label in enumerate(lab_list, 1)}

    train_features = convert_examples_to_features(train_examples, max_seq_length, tokenizer)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", trainBatchSize)
    logger.info("  Num steps = %d", numTrainOptimizationSteps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=trainBatchSize)

    model.train()

    train_losses = []
    best_losses = 1
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    epoch = 0

    for _ in trange(int(nIterations), desc="Epoch"):
        epoch += 1
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = batch
            if modelType in ["bert-base-cased", "allenai/scibert_scivocab_cased", "distilbert-base-cased"]:
                loss = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask)
            else:
                loss = model(input_ids, input_mask, label_ids, valid_ids, l_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), maxGradNorm)

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if (step + 1) % 1 == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

        tr_losses = tr_loss / len(train_dataloader)
        last_model = model.module if hasattr(model, 'module') else model

        if tr_losses < best_losses or 'model_to_save' not in locals():
            print("Epoch -" + str(epoch) + ": Train losses - " + str(tr_losses) + " / Best losses - " + str(best_losses))
            best_losses = tr_losses
            model_to_save = model.module if hasattr(model, 'module') else model

        train_losses.append(tr_losses)
        print(train_losses)
        print(len(train_losses))

    model = model_to_save

    # Save a trained model and the associated configuration
    print(train_losses)
    print(len(train_losses))
    model_to_save.save_pretrained(modelPath)
    tokenizer.save_pretrained(modelPath)
    label_map = {i: label for i, label in enumerate(lab_list, 1)}

    model_config = {"bert_model": modelType,
                    "do_lower": do_lower_case,
                    "max_seq_length": max_seq_length,
                    "num_labels": len(lab_list) + 1,
                    "label_map": label_map}

    json.dump(model_config, open(os.path.join(modelPath, "model_config.json"), "w"))
    plt.plot(train_losses, '-o')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Train'])
    plt.title('Train Loss')
        
    shutil.copyfile(os.path.abspath("train.txt"),modelPath+"train.txt")
    shutil.copyfile(os.path.abspath("valid.txt"),modelPath+"valid.txt")
    plt.savefig(modelPath + '/losses.png')
    plt.clf()

    last_model.save_pretrained(modelPath + "/last")

    if (useCuda):
        model.cuda()

    model.to(device)


def test(modelType, modelPath):
    evalBatchSize = 8

    task_name = "ner"
    processors = {task_name: NerProcessor}
    processor = processors[task_name]()

    model = Nertrain.from_pretrained(modelPath)
    tokenizer = BertTokenizer.from_pretrained(modelPath, do_lower_case=do_lower_case)

    eval_examples = processor.get_test_examples("")
    eval_features = convert_examples_to_features(eval_examples, lab_list, max_seq_length, tokenizer)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", evalBatchSize)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                              all_lmask_ids)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=evalBatchSize)
    model.eval()
    y_true = []
    y_pred = []
    label_map = {i: label for i, label in enumerate(lab_list, 1)}

    for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        valid_ids = valid_ids.to(device)
        label_ids = label_ids.to(device)
        l_mask = l_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids, attention_mask_label=l_mask)

        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == len(label_map):
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                else:
                    try:
                        temp_1.append(label_map[label_ids[i][j]])
                    except:
                        print(i)
                        print(j)
                        print(label_ids[i][j])

                    lab_pred = logits[i][j]
                    if "CLS" in label_map[lab_pred] or "SEP" in label_map[lab_pred]:
                        lab_pred = 1

                    temp_2.append(label_map[lab_pred])

    report = classification_report(y_true, y_pred, digits=4)
    flat_y_true = [i for j in y_true for i in j]
    flat_y_pred = [i for j in y_pred for i in j]
    cf_matrix.generate_confusion_matrix(flat_y_true, flat_y_pred, modelPath, modelType)

    logger.info("\n%s", report)
    output_eval_file = os.path.join(modelPath, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        logger.info("\n%s", report)

        writer.write("***** Eval results *****")
        writer.write(report)
