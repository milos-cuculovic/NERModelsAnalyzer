"""
Created on Fri Jun 25 17:49:48 2021
@author: chams
"""
import shutil
import os
from torch import nn
import json
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from pytorch_transformers import (AdamW,
                                  BertForTokenClassification, BertTokenizer,
                                  WarmupLinearSchedule)
from transformers import pipeline, AutoModelForTokenClassification
from transformers import BertConfig, BertForTokenClassification
from transformers import pipeline, AutoModelForTokenClassification
from transformers import BertTokenizer

import torch
from tqdm import tqdm, trange
from seqeval.metrics import classification_report
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from nltk import word_tokenize
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from bertconf import removEsc, sentenceMean, json_conll, trigConll, crossval, changeToOther
import shutil
from grid_search_results_print import generate_grid_search_results_print

trigger= ['why', 'on the contrary','what','however','either','while','rather','instead of', 'when','than',
         'in order to','therefore','not only', 'afterwards','once again','or','in order to','in particular',
         'also','if not','if not then','and','not only','does','albeit','because','is that','that','without','who',
         'whether','is it', 'was it','such as','were they','are they','thus','again','given that','given the',
         'how many','except','nor','both','whose','especialls','for instance','is this','similarly','were there',
         'are there','is there','for the time being','based on','in particular','as currently','perhaps','once',
         'how','otherwise','particularly','overall','although','prior to','At the same time',
         'neither','apart from','besides from','if necessary','hence','how much','by doing so','since','how less'
         'despite','accordingly','etc','always','what kind','unless','which one','if not','if so','even if',
         'not just','not only','besides','after all','generally','similar to','too','like']

labelremove=["CONTENT"]
lab_list=["O", "B-LOCATION", "I-LOCATION", "B-TRIGGER", "I-TRIGGER",
                "B-MODAL", "I-MODAL", "B-ACTION", "I-ACTION", "B-CONTENT", "I-CONTENT", "[CLS]", "[SEP]"]


class Nertrain(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None):
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


class BertNer(BertForTokenClassification):
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(batch_size):
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

device = 'cpu'




def trainBERTModel(jsonfile, output_dir, nIter, use_cuda):

    learning_rate       = 2e-05
    weight_decay        = 0.001
    warmup_proportion   = 0.1
    train_batch_size    = 26

    # # # INITIAL
    # removEsc(os.path.abspath(jsonfile))

    # # STEP ONE cross validation
    # crossval(os.path.abspath(jsonfile), os.path.abspath(""))

    # # STEP TWO remove sentence without action and location
    #sentenceMean(os.path.abspath("train1.json"))

    # # STEP THREE convert json to conll
    # json_conll(os.path.abspath("train1.json"), os.path.abspath(""), 'train_final_1.txt')
    # json_conll(os.path.abspath("valid1.json"), os.path.abspath(""), 'valid_final_1.txt')
    # # # STEP FOUR REPLACE TRIGGER
    # trigConll(os.path.abspath("train_final_1.txt"), trigger)
    # trigConll(os.path.abspath("valid_final_1.txt"), trigger)

    global device
    if use_cuda == True:
        device = "cuda"
    else:
        device = "cpu"
    
    shutil.copyfile(r'train.txt', r'train_temp.txt')
    shutil.copyfile(r'valid.txt', r'valid_temp.txt')
       
    for i in labelremove:
            print("here")
            changeToOther(i,"train_temp.txt")
            changeToOther(i,"valid_temp.txt")
            lab_list.remove("I-"+i)
            lab_list.remove("B-"+i)
           
    trainBert(output_dir, train_batch_size, True, int(nIter), use_cuda, True, 1, learning_rate,
                weight_decay, warmup_proportion)
    os.remove("train_temp.txt")
    os.remove("valid_temp.txt")
    
def trainBERTGrid(jsonfile, output_dir, nIter, use_cuda):
    # # INITIAL
    # removEsc(os.path.abspath(jsonfile))

    # # STEP ONE cross validation
    # crossval(os.path.abspath(jsonfile), os.path.abspath(""))

    # # STEP TWO remove sentence without action and location
    # sentenceMean(os.path.abspath("train1.json"))

    # # STEP THREE convert json to conll
    # json_conll(os.path.abspath("train1.json"), os.path.abspath(""), 'train.txt')
    # json_conll(os.path.abspath("valid1.json"), os.path.abspath(""), 'valid.txt')

    # # STEP FOUR REPLACE TRIGGER
    # trigConll(os.path.abspath("train.txt"), trigger)
    # trigConll(os.path.abspath("valid.txt"), trigger)

    global device
    if use_cuda == True:
        device = "cuda"
    else:
        device = "cpu"

    loopBerthyperparam(output_dir, int(nIter), use_cuda)

def evaluation(output_dir, use_cuda):
    
    shutil.copyfile(r'train.txt', r'train_temp.txt')
    
    shutil.copyfile(r'valid.txt', r'valid_temp.txt')
    for i in labelremove:
        changeToOther(i,"train_temp.txt")
        changeToOther(i,"valid_temp.txt")
        lab_list.remove("I-"+i)
        lab_list.remove("B-"+i)
    trainBert(output_dir, 32, False, 1, use_cuda, True,"",2e-5,0.01,0.1)
    os.remove("train_temp.txt")
    os.remove("valid_temp.txt")

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
    '''
    read file
    '''
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
        sentence = []
        label = []
    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

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
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return lab_list

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


local_rank = -1
fp16 = 'store_true'
gradient_accumulation_steps = 1
seed = 42
eval_batch_size = 8
bert_model = "bert-base-cased"
adam_epsilon = 1e-8
max_grad_norm = 1.0
max_seq_length = 128
do_lower_case = "store_false"
fp16_opt_level = 'O1'
b1 = 0.9
b2 = 0.999
import itertools

def loopBerthyperparam(output_dir,num_train_epochs,use_cuda):
    weightdecay         = [0.1, 0.01, 0.001, 0.0001]
    learningrate        = [2e-5, 2.2e-5, 2.4e-5, 2.6e-5, 2.8e-5, 3e-5]
    warmupproportion    = [0.1]
    trainbatchsize      = [32, 30, 28, 26, 24, 22, 20, 18, 16]
    hyperparam          = [weightdecay, learningrate, warmupproportion, trainbatchsize]
    k                   = 0

    list_permutations = list(itertools.product(*hyperparam))
    shutil.copyfile(r'train.txt', r'train_temp.txt')
    
    shutil.copyfile(r'valid.txt', r'valid_temp.txt')
    for i in labelremove:
        changeToOther(i,"train_temp.txt")
        changeToOther(i,"valid_temp.txt")
        lab_list.remove("I-"+i)
        lab_list.remove("B-"+i)
    for listtool in list_permutations:
        k += 1
        weight = listtool[0]
        learning = listtool[1]
        warm = listtool[2]
        trainbs = listtool[3]

        #trainBert(output_dir, trainbs, True, num_train_epochs, use_cuda, True, k, learning, weight, warm)

    os.remove("train_temp.txt")
    os.remove("valid_temp.txt")
    compareauto(list_permutations, output_dir)

def compareauto(list_permutations,filename):
    results = {}
    precision_loc = [0, 0]
    recall_loc = [0, 0]
    f1score_loc = [0, 0]
    precision_wght = [0, 0]
    recall_wght = [0, 0]
    f1score_wght = [0, 0]
    grid_search = {}

    for i in range(1, len(list_permutations) + 1):
        with open(filename + str(i) + "/eval_results.txt") as file:
            for line in file:
                line[0].split()
                for line in file:
                    listword = line.split()
                    if len(listword) > 0:
                        if listword[0] == "LOCATION":
                            precision_loc, recall_loc, f1score_loc \
                                = get_best_grid_scores(precision_loc, recall_loc, f1score_loc, listword, i)
                            results['LOCATION'] = [precision_loc, recall_loc, f1score_loc]
                            weightdecay = list_permutations[i][0]
                            learningrate = list_permutations[i][1]
                            trainbatchsize = list_permutations[i][3]


                            grid_search[i] = [weightdecay, learningrate, trainbatchsize, f1score_loc[1]]
                        if listword[0] == "weighted":
                            precision_wght, recall_wght, f1score_wght\
                                = get_best_grid_scores(precision_wght, recall_wght, f1score_wght, listword[1:], i)
                            results['weighted'] = [precision_wght, recall_wght, f1score_wght]

    for result in results:
        print(result)
        print("   precision n " + str(results[result][0][0]) + " - " + str(results[result][0][1]))
        print("   recall n " + str(results[result][1][0]) + " - " + str(results[result][1][1]))
        print("   f1score n " + str(results[result][2][0]) + " - " + str(results[result][2][1]))

    #generate_grid_search_results_print(grid_search)


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
                  
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

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
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

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
        # # grouped_entities=True,
        aggregation_strategy="SIMPLE",
        model=mode,
        tokenizer=tokenize
    )
    nlp_ner.save_pretrained(new_model_path)


def prediction(t, model_name):
    model_path = os.path.dirname(os.path.abspath(__file__)) + '/trained_models/' + model_name
    mode = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenize = BertTokenizer.from_pretrained(model_path)
    # mode("test")
    nlp_ner = pipeline(
        "ner",
        model=mode,
        tokenizer=tokenize
    )
    label_list = ["O", "B-LOCATION", "I-LOCATION", "B-TRIGGER", "I-TRIGGER",
                  "B-MODAL", "I-MODAL", "B-ACTION", "I-ACTION",  "[CLS]", "[SEP]"]

    #label_list = ["B-LOCATION", "I-LOCATION", "B-TRIGGER", "I-TRIGGER",
    #              "B-MODAL", "I-MODAL", "B-ACTION", "I-ACTION", "B-CONTENT", "I-CONTENT"]
    print(nlp_ner(t))
    prediction = []
    initial = True
    dicINT = {}
    result = nlp_ner(t)

    for dic in result:
        label = dic['entity']
        index = label.find("_") + 1
        number = label[index:]
        pos = int(number) - 1
        label_name = label_list[pos]
        word = dic['word']


        if label_name in ("O", "[CLS]", "[SEP]"):
            continue

        if label_name[2:] in dicINT:
            if "##" in word:
                word = word.lstrip('##')
                dicINT[label_name[2:]] += word
            elif word == "'" or word == "s":
                dicINT[label_name[2:]] += word
            else:
                dicINT[label_name[2:]] += " " + dic['word']
        else:
            if initial == False:
                prediction.append(dicINT)
                dicINT = {}
            dicINT[label_name[2:]] = dic['word']

        initial = False


    prediction.append(dicINT)

    return prediction





def trainBert(output_dir, train_batch_size, do_train, num_train_epochs, use_cuda, do_eval, indexEval,
              learning_rate, weight_decay, warmup_proportion):
    processors = {"ner": NerProcessor}

    n_gpu = torch.cuda.device_count()

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(local_rank != -1), fp16))

    if gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            gradient_accumulation_steps))

    train_batch_size = train_batch_size // gradient_accumulation_steps

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_dir=output_dir+str(indexEval)
    if os.path.exists(output_dir) and os.listdir(output_dir) and do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    task_name = "ner".lower()

    processor = processors[task_name]()
    label_list =lab_list
    num_labels = len(label_list) + 1

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    train_examples = None
    num_train_optimization_steps = 0.0
    if do_train:
        train_examples = processor.get_train_examples("")
        num_train_optimization_steps = int(
            len(train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
        if local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Prepare model
    config = BertConfig.from_pretrained(bert_model, num_labels=num_labels, finetuning_task=task_name)
    model = Nertrain.from_pretrained(bert_model, from_tf=False, config=config)

    if local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if (use_cuda):
        model.cuda()

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, betas=(b1, b2), eps=adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    if do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                                   all_lmask_ids)
        if local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=
        train_batch_size)

        model.train()
        train_losses = []
        best_losses = 1

        for _ in trange(int(num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                # if fp16:
                #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

            tr_losses = tr_loss / len(train_dataloader)

            if tr_losses < best_losses:
                best_losses = tr_losses
                model_to_save = model.module if hasattr(model, 'module') else model

            if tr_losses < 0.05:
                break
            train_losses.append(tr_losses)
            print(train_losses)
            print(len(train_losses))

        # Save a trained model and the associated configuration
        print(train_losses)
        print(len(train_losses))
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        model_config = {"bert_model": bert_model, "do_lower": do_lower_case,
                        "max_seq_length": max_seq_length, "num_labels": len(label_list) + 1,
                        "label_map": label_map}
        json.dump(model_config, open(os.path.join(output_dir, "model_config.json"), "w"))
        plt.plot(train_losses, '-o')
        # plt.plot(eval_accu,'-o')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Train'])
        plt.title('Train Loss')
        
        shutil.copyfile(os.path.abspath("train.txt"),output_dir+"train.txt")  
        shutil.copyfile(os.path.abspath("valid.txt"),output_dir+"valid.txt")
        plt.savefig(output_dir + '/losses.png')
        
    else:
        # Load a trained model and vocabulary that you have fine-tuned
        model = Nertrain.from_pretrained(output_dir)
        tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=do_lower_case)

    if (use_cuda):
        model.cuda()
    model.to(device)

    if do_eval:
        
        eval_examples = processor.get_dev_examples("")
        eval_features = convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                                  all_lmask_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
        model.eval()
        y_true = []
        y_pred = []
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(eval_dataloader,
                                                                                     desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)
            # print(label_map)
            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids, attention_mask_label=l_mask)
            # print(label_ids)
            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()
            # print(logits[1])
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            # print(label_map)
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

                        lab_pred=logits[i][j]
                        temp_2.append(label_map[lab_pred])
                        if "CLS" in label_map[lab_pred] or  "SEP" in label_map[lab_pred]:
                            lab_pred=1
                        
        
        report = classification_report(y_true, y_pred, digits=4)
        flat_y_true = [i for j in y_true for i in j]
        flat_y_pred = [i for j in y_pred for i in j]
        # cf_matrix.generate_plotly_cf_mat(flat_y_true, flat_y_pred, label_map, "confusion_matrix.html",  "./visualizations/")
        logger.info("\n%s", report)
        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("weight_decay:" + str(weight_decay))
            logger.info("learning_rate:" + str(learning_rate))
            logger.info("warmup:" + str(warmup_proportion))
            logger.info("train batch size:" + str(train_batch_size))
            logger.info("\n%s", report)

            writer.write("***** Eval results *****")
            writer.write("weight_decay:" + str(weight_decay))
            writer.write("learning_rate:" + str(learning_rate))
            writer.write("warmup:" + str(warmup_proportion))
            writer.write("train batch size:" + str(train_batch_size))
            writer.write(report)
