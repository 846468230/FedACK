# -*- coding: utf-8 -*-

import random
from collections import namedtuple
from typing import Dict
import numpy as np
import torch

from data.field import Field

Batch = namedtuple("Batch", ['src', 'tgt', 'batch_size'])
Example = namedtuple("Example", ['src', 'tgt'])


class TranslationDataset(object):

    def __init__(self,
                 src_path: str,
                 tgt_path: str,
                 batch_size: int,
                 device: torch.device,
                 train: bool,
                 fields: Dict[str, Field]):

        self.batch_size = batch_size
        self.train = train
        self.device = device
        self.fields = fields
        self.sort_key = lambda ex: (len(ex.src), len(ex.tgt))

        examples = []
        for src_line, tgt_line in zip(read_file(src_path), read_file(tgt_path)):
            examples.append(Example(src_line, tgt_line))
        if self.train:
            examples, self.seed = self.sort(examples)

        self.num_examples = len(examples)
        self.batches = list(batch(examples, self.batch_size))

    def __iter__(self):
        while True:
            if self.train:
                random.shuffle(self.batches)
            for minibatch,curlen in self.batches:
                src = self.fields["src"].process([x.src for x in minibatch], self.device,curlen)   #minibatch里所有的padding后的句子的索引矩阵。
                tgt = self.fields["tgt"].process([x.tgt for x in minibatch], self.device,curlen)
                yield Batch(src=src, tgt=tgt, batch_size=len(minibatch)) # 词的索引矩阵
            if not self.train:
                break

    def sort(self, examples):
        seed = sorted(range(len(examples)), key=lambda idx: self.sort_key(examples[idx]))
        return sorted(examples, key=self.sort_key), seed


class TestDataSet(TranslationDataset):
    def __init__(self,
                 src_path: str,
                 tgt_path: str,
                 batch_size: int,
                 device: torch.device,
                 train: bool,
                 fields: Dict[str, Field],V=50,num_examples=1000):
        self.batch_size = batch_size
        self.train = train
        self.device = device
        self.fields = fields
        self.num_examples = num_examples
        if not train:
            self.num_examples = 1000
        self.sort_key = lambda ex: (len(ex.src), len(ex.tgt))
        examples = []
        for i in range(self.num_examples):     # splited list
            src = list(np.random.randint(1,V,size=(20,)))
            src = [str(x) for x in src]
            tgt = src[:]
            examples.append(Example(src,tgt))
        if self.train:
            examples, self.seed = self.sort(examples)

        self.batches = list(batch(examples, self.batch_size))


def read_file(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.strip().split()


def batch(data, batch_size):
    minibatch, cur_len = [], 0
    for ex in data:
        minibatch.append(ex)
        cur_len = max(cur_len, len(ex.src), len(ex.tgt))
        if cur_len * len(minibatch) > batch_size:
            yield (minibatch[:-1],cur_len)
            minibatch, cur_len = [ex], max(len(ex.src), len(ex.tgt))
    if minibatch:
        yield (minibatch,cur_len)