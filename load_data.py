from torch.utils.data import Dataset
from multiprocessing import Pool
import os
import random
import pickle
import torch
from newsroom import jsonl
from nltk import sent_tokenize


def write_pkl(object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)


def load_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def pad_to_max_len(input_ids, labels, masks=None):
    max_len = max([max([len(s) for s in input_id]) for input_id in input_ids])
    max_doc_len = max(len(input_id) for input_id in input_ids)
    new_ids = []
    new_labels = []
    real_lengths = []
    for input_id, label in zip(input_ids, labels):
        sents = [s + [0]*(max_len-len(s)) for s in input_id]
        real_length = len(input_id)
        sents += [[0]*max_len] * (max_doc_len-real_length)
        label += [0] * (max_doc_len-real_length)
        new_ids.append(sents)
        new_labels.append(label)
        real_lengths.append(real_length)
    # masks = torch.tensor([[1]*len(input_id)+[0]*(max_len-len(input_id)) for input_id in input_ids], dtype=torch.long)
    # input_ids = torch.tensor([input_id+[0]*(max_len-len(input_id)) for input_id in input_ids], dtype=torch.long)
    # return input_ids, masks
    new_ids = torch.tensor(new_ids, dtype=torch.long)
    new_labels = torch.tensor(new_labels, dtype=torch.long)
    real_lengths = torch.tensor(real_lengths, dtype=torch.long)
    return new_ids, new_labels, real_lengths


def _pad_to_max_len(input_ids):
    max_len = max([max([len(s) for s in input_id]) for input_id in input_ids])
    max_doc_len = max(len(input_id) for input_id in input_ids)
    new_ids = []
    real_lengths = []
    for input_id in input_ids:
        sents = [s + [0]*(max_len-len(s)) for s in input_id]
        real_length = len(input_id)
        sents += [[0]*max_len] * (max_doc_len-real_length)
        new_ids.append(sents)
        real_lengths.append(real_length)
    new_ids = torch.tensor(new_ids, dtype=torch.long)
    real_lengths = torch.tensor(real_lengths, dtype=torch.long)
    return new_ids, real_lengths


def dynamic_collate_fn(batch):
    input_ids, labels = list(zip(*batch))
    input_ids, labels, real_lengths = pad_to_max_len(input_ids, labels)
    return input_ids, labels, real_lengths


def _dynamic_collate_fn(batch):
    input_ids, doc, summary = list(zip(*batch))
    input_ids, real_lengths = _pad_to_max_len(input_ids)
    return input_ids, doc, summary, real_lengths


def ner_pad_to_max_len(input_ids, labels):
    real_lengths = [len(s) for s in input_ids]
    max_len = max(real_lengths)
    # masks = torch.tensor([[1]*len(input_id)+[0]*(max_len-len(input_id)) for input_id in input_ids], dtype=torch.long)
    input_ids = torch.tensor([input_id+[0]*(max_len-len(input_id)) for input_id in input_ids], dtype=torch.long)
    labels = torch.tensor([label+[0]*(max_len-len(label)) for label in labels], dtype=torch.long)
    # return input_ids, masks
    return input_ids, labels, real_lengths


def ner_collate_fn(batch):
    input_ids, labels = list(zip(*batch))
    input_ids, labels, real_lengths = ner_pad_to_max_len(input_ids, labels)
    return input_ids, labels, real_lengths


class TextDataset(Dataset):
    def __init__(self, task_id, mode, args, tokenizer):
        self.task_id = task_id
        self.task = args.tasks[task_id]
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.max_doc_len = args.max_doc_len

        if self.mode == 'train':
            self.data = load_pkl("./data/newsroom/{}_{}.pkl".format(self.task, mode))
            random.shuffle(self.data)
            if args.ratio != 1.0:
                self.data = self.data[:int(args.ratio * len(self.data))]
            with Pool(args.n_workers) as pool:
                self.data = pool.map(self.map_train_data, self.data)
        elif self.mode == 'test':
            self.data = []
            with jsonl.open('./data/newsroom/test.jsonl.gz', gzip=True) as train_file:
                for entry in train_file:
                    url = entry['url'].split('/')[2].split('.')
                    if self.task in url:
                        self.data.append((entry['text'], entry['summary']))
            if args.ratio != 1.0:
                self.data = self.data[:int(args.ratio * len(self.data))]
            with Pool(args.n_workers) as pool:
                self.data = pool.map(self.map_test_data, self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def map_train_data(self, doc_and_labels):
        doc, labels = doc_and_labels
        gold_labels = [1 if i in labels else 0 for i in range(len(doc[:self.max_doc_len]))]
        input_ids = [self.tokenizer.encode(sent[:self.max_len]) for sent in doc[:self.max_doc_len]]
        return input_ids, gold_labels

    def map_test_data(self, datas):
        text, summary = datas
        doc = [sent for sent in sent_tokenize(text) if len(sent.replace('.', '')) >= 1]
        input_ids = [self.tokenizer.encode(sent[:self.max_len]) for sent in doc[:self.max_doc_len]]
        return input_ids, doc[:self.max_doc_len], summary


class NERDataset(Dataset):
    def __init__(self, task, mode, args, tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        label_map = {'O': 0, 'ORG': 1, 'LOC': 2, 'PER': 3, 'MISC': 4, 'PERSON': 3}
        if task == 'CoNLL2003':
            with open('./data/CoNLL2003/{}.txt'.format(mode)) as f:
                texts = f.read().split('\n\n')
                for text in texts:
                    words, labels = [], []
                    for line in text.split('\n'):
                        if '-DOCSTART-' in line or not line:
                            continue
                        items = line.split()
                        words.append(items[0])
                        labels.append(label_map[items[-1].split('-')[-1]])
                    if words and labels:
                        self.data.append((words, labels))
        elif task == 'OntoNotes':
            with open('./data/OntoNotes/onto.{}.ner'.format(mode)) as f:
                texts = f.read().strip('\n').split('\n\n')
                for text in texts:
                    words, labels = [], []
                    for line in text.split('\n'):
                        items = line.split()
                        if not items:
                            continue
                        words.append(items[0])
                        labels.append(label_map.get(items[-1].split('-')[-1], 4))
                    if words and labels:
                        self.data.append((words, labels))

        if args.ratio != 1.0:
            self.data = self.data[:int(args.ratio * len(self.data))]
        with Pool(args.n_workers) as pool:
            self.data = pool.map(self.map_data, self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def map_data(self, data_and_labels):
        data, labels = data_and_labels
        input_ids = self.tokenizer.encode(' '.join(data))
        return input_ids, labels


class RNN_tokenizer:
    def __init__(self, args):
        with open(args.vocab_path, 'r', encoding='utf-8') as f:
            self.vocabs = f.read().strip('\n').split('\n')
            self.vocab2idx = {word: idx for idx, word in enumerate(self.vocabs)}

    def encode(self, context):
        return [self.vocab2idx.get(word, 0) for word in context.split()]

    def decode(self, ids, labels, pred):
        return ' '.join([self.vocabs[input_id] + '/' + str(label) + '/' + str(p)
                         for input_id, label, p in zip(ids, labels, pred)])
