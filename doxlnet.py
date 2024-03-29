"""
Created on Tue May  3 15:33:04 2022
@author: chams
refactored by Milos - 07.09.2022
"""

import os
import json
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from pytorch_transformers import (AdamW,WarmupLinearSchedule)
from transformers import XLNetTokenizer, XLNetForTokenClassification, XLNetConfig
from transformers import pipeline, AutoModelForTokenClassification
import cf_matrix
import torch
from tqdm import tqdm, trange
from seqeval.metrics import classification_report
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from nltk import word_tokenize
import torch.distributed as dist
from bertconf import removEsc, sentenceMean, json_conll, trigConll, crossval, changeToOther
import shutil
from grid_search_results_print import print_3D_graph

device = 'cpu'
labelremove=["CONTENT"]
lab_list=["O", "B-LOCATION", "I-LOCATION", "B-TRIGGER", "I-TRIGGER",
                "B-MODAL", "I-MODAL", "B-ACTION", "I-ACTION", "B-CONTENT", "I-CONTENT"]

class Ner:
    def __init__(self, model_dir: str):
        self.model, self.tokenizer, self.model_config = self.load_model(model_dir)
        self.label_map = self.model_config["label_map"]
        self.max_seq_length = self.model_config["max_seq_length"]
        self.label_map = {int(k): v for k, v in self.label_map.items()}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_dir: str, model_config: str = "model_config.json"):
        model_config = os.path.join(model_dir, model_config)
        model_config = json.load(open(model_config))
        model = XLNetForTokenClassification.from_pretrained(model_dir)
        tokenizer = XLNetTokenizer.from_pretrained(model_dir, do_lower_case=model_config["do_lower"],
                                                   return_tensors="pt")
        return model, tokenizer, model_config

    def tokenize(self, text: str):
        """ tokenize input"""
        words = word_tokenize(text)
        tokens = []
        valid_positions = []
        for i, word in enumerate(words):
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
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            valid_positions.append(0)
        return input_ids, input_mask, valid_positions

    def predict(self, text: str):
        logits = []
        logits.pop()

        labels = [(self.label_map[label], confidence) for label, confidence in logits]
        words = word_tokenize(text)
        assert len(labels) == len(words)
        output = [{"word": word, "tag": label, "confidence": confidence} for word, (label, confidence) in
                  zip(words, labels)]
        return output


def train_xlnet_model(output_dir, nIter, use_cuda):
    learning_rate = 0.00025
    weight_decay = 0.01
    warmup_proportion = 0.1
    train_batch_size = 12

    # INITIAL
    #removEsc(os.path.abspath(jsonfile))

    # STEP ONE cross validation
    # crossval(os.path.abspath(jsonfile), os.path.abspath(""))

    # STEP TWO remove sentence without action and location
    # sentenceMean(os.path.abspath("train.json"))

    # STEP THREE convert json to conll
    # json_conll(os.path.abspath("train.json"), os.path.abspath(""), 'train.txt')
    # json_conll(os.path.abspath("valid.json"), os.path.abspath(""), 'valid.txt')

    # STEP FOUR REPLACE TRIGGER
    # trigConll(os.path.abspath("train.txt"), trigger)
    # trigConll(os.path.abspath("valid.txt"), trigger)

    global device
    if use_cuda == True:
        device = "cuda"
    else:
        device = "cpu"

    trainxlnet(output_dir, train_batch_size, False, int(nIter), use_cuda, True, 0, learning_rate,
               weight_decay, warmup_proportion)


def grid_xlnet_model(jsonfile, output_dir, nIter, use_cuda):
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

    loopxlnethyperparam(output_dir, int(nIter), use_cuda)


def test_xlnet_model(output_dir, use_cuda):
    trainxlnet(output_dir, 32, False, 1, use_cuda, True, "", 2e-5, 0.01, 0.1)


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
xlnet_model = "xlnet-base-cased"
adam_epsilon = 1e-8
max_grad_norm = 1.0
max_seq_length = 128
do_lower_case = "store_false"
fp16_opt_level = 'O1'
b1 = 0.9
b2 = 0.999
import itertools


def loopxlnethyperparam(output_dir, num_train_epochs, use_cuda):
    weightdecay = [0.1, 0.01, 0.001, 0.0001]
    learningrate = [0.01, 0.001, 0.0001, 0.00001]
    warmupproportion = [0.1]
    trainbatchsize = [16, 32, 64, 96]

    hyperparam = [weightdecay, learningrate, warmupproportion, trainbatchsize]
    k = 0

    list_permutations = list(itertools.product(*hyperparam))
    shutil.copyfile(r'train.txt', r'train_temp.txt')
    shutil.copyfile(r'valid.txt', r'valid_temp.txt')

    for i in labelremove:
        changeToOther(i, "train_temp.txt")
        changeToOther(i, "valid_temp.txt")
        lab_list.remove("I-" + i)
        lab_list.remove("B-" + i)
    for listtool in list_permutations:
        k += 1
        weight = listtool[0]
        learning = listtool[1]
        warm = listtool[2]
        trainbs = listtool[3]

        trainxlnet(output_dir, trainbs, False, num_train_epochs, use_cuda, True, k, learning, weight, warm)

    compareauto(list_permutations, output_dir)

    os.remove("train_temp.txt")
    os.remove("valid_temp.txt")


def compareauto(list_permutations,output_dir):
    results = {}
    precision_loc = [0, 0]
    recall_loc = [0, 0]
    f1score_loc = [0, 0]
    precision_wght = [0, 0]
    recall_wght = [0, 0]
    f1score_wght = [0, 0]
    grid_search = {}

    for i in range(0, len(list_permutations)):
        with open(output_dir + "/" + str(i+1) + "/eval_results.txt") as file:
            for line in file:
                line[0].split()
                for line in file:
                    listword = line.split()
                    if len(listword) > 0:
                        if listword[0] == "LOCATION":
                            precision_loc, recall_loc, f1score_loc \
                                = get_best_grid_scores(precision_loc, recall_loc, f1score_loc, listword, i+1)
                            results['LOCATION'] = [precision_loc, recall_loc, f1score_loc]

                        if listword[0] == "weighted":
                            precision_wght, recall_wght, f1score_wght\
                                = get_best_grid_scores(precision_wght, recall_wght, f1score_wght, listword[1:], i+1)
                            results['weighted'] = [precision_wght, recall_wght, f1score_wght]
                            weightdecay = list_permutations[i][0]
                            learningrate = list_permutations[i][1]
                            trainbatchsize = list_permutations[i][3]
                            grid_search[i] = [weightdecay, learningrate, trainbatchsize, listword[4]]

    for result in results:
        print("   precision n " + str(results[result][0][0]) + " - " + str(results[result][0][1]))
        print("   recall n " + str(results[result][1][0]) + " - " + str(results[result][1][1]))
        print("   f1score n " + str(results[result][2][0]) + " - " + str(results[result][2][1]))

    generate_grid_search_results_print(grid_search, output_dir, xlnet_model)


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
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
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
    tokenize = XLNetTokenizer.from_pretrained(model_path)
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
    tokenize = XLNetTokenizer.from_pretrained(model_path)
    # mode("test")
    nlp_ner = pipeline(
        "ner",
        model=mode,
        tokenizer=tokenize
    )
    label_list = ["O", "B-LOCATION", "I-LOCATION", "B-TRIGGER", "I-TRIGGER",
                  "B-MODAL", "I-MODAL", "B-ACTION", "I-ACTION", "B-CONTENT", "I-CONTENT"]

    # label_list = ["B-LOCATION", "I-LOCATION", "B-TRIGGER", "I-TRIGGER",
    #              "B-MODAL", "I-MODAL", "B-ACTION", "I-ACTION", "B-CONTENT", "I-CONTENT"]
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

        if word == "'":
            label_name = "I-CONTENT"

        if label_name in ("O"):
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


def trainxlnet(output_dir, train_batch_size, do_train, num_train_epochs, use_cuda, do_eval, indexEval,
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
    if indexEval != 0:
        output_dir = output_dir + str(indexEval)
    if os.path.exists(output_dir) and os.listdir(output_dir) and do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    task_name = "ner".lower()

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    tokenizer = XLNetTokenizer.from_pretrained(xlnet_model, do_lower_case=do_lower_case, return_tensors="pt")

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
    config = XLNetConfig.from_pretrained(xlnet_model, num_labels=num_labels, finetuning_task=task_name)
    model = XLNetForTokenClassification.from_pretrained(xlnet_model, from_tf=False, config=config)

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
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids,
                                   all_lmask_ids)
        if local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=
        train_batch_size)

        model.train()
        train_losses = []

        for _ in trange(int(num_train_epochs), desc="Epoch"):
            tr_loss = 0
            best_losses=1
            nb_tr_examples, nb_tr_steps = 0, 0
            """          for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, label_ids, l_mask = batch
                loss = model(input_ids=input_ids, attention_mask=input_mask,
                               attention_mask_label=l_mask, labels=label_ids)
                loss = loss["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad() # added
                model.zero_grad()
                global_step+=1
                tr_loss += loss.item()"""
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, label_ids, l_mask = batch
                loss = model(input_ids=input_ids, attention_mask=input_mask,
                               attention_mask_label=l_mask, labels=label_ids)
                loss = loss["loss"]                
                loss = loss.mean()  # mean() to average on multi-gpu.
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

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

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        model_config = {"XLNet_model": xlnet_model, "do_lower": do_lower_case,
                        "max_seq_length": max_seq_length, "num_labels": len(label_list) + 1,
                        "label_map": label_map}
        json.dump(model_config, open(os.path.join(output_dir, "model_config.json"), "w"))
        plt.plot(train_losses, '-o')
        # plt.plot(eval_accu,'-o')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Train'])
        plt.title('Train Loss')

        plt.show()

        plt.savefig(output_dir + '/losses.png')
    else:
        # Load a trained model and vocabulary that you have fine-tuned
        model = XLNetForTokenClassification.from_pretrained(output_dir)
        tokenizer = XLNetTokenizer.from_pretrained(output_dir, do_lower_case=do_lower_case)

    if (use_cuda):
        model.cuda()
    model.to(device)

    if do_eval and (local_rank == -1 or torch.distributed.get_rank() == 0):

        eval_examples = processor.get_dev_examples("")
        eval_features = convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids,
                                  all_lmask_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
        model.eval()
        y_true = []
        y_pred = []
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        for input_ids, input_mask, label_ids, l_mask in tqdm(eval_dataloader,
                                                             desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)

            with torch.no_grad():
                outputs = model(input_ids, input_mask, attention_mask_label=l_mask)
            logits = outputs.logits
            softmax = F.softmax(logits, dim=2)
            index = torch.argmax(softmax, dim=2)
            index = index.detach().cpu().numpy()
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

                    elif label_ids[i][j] == 0:
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j]])

                        lab_pred = index[i][j]
                        if lab_pred == 0:
                            lab_pred = 1
                        temp_2.append(label_map[lab_pred])
        report = classification_report(y_true, y_pred, digits=4)
        flat_y_true = [i for j in y_true for i in j]
        flat_y_pred = [i for j in y_pred for i in j]

        cf_matrix.generate_confusion_matrix(flat_y_true, flat_y_pred, output_dir, xlnet_model)
        logger.info("\n%s", report)
        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("weight_decay:" + str(weight_decay))
            logger.info("learning_rate:" + str(learning_rate))
            logger.info("warmup:" + str(warmup_proportion))
            logger.info("train batch size:" + str(train_batch_size))
            logger.info("\n%s", report)
            writer.write(report)