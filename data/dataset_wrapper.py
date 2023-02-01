# -*- coding: utf-8 -*-
import itertools

from data.dataset import TranslationDataset


# from beaver.data.dataset_sum import SumTransDataset

class Dataset(object):
    def __init__(self, task1_dataset: TranslationDataset):
        self.task1_dataset = task1_dataset

        self.fields = {
            "src": task1_dataset.fields["src"],
            "task1_tgt": task1_dataset.fields["tgt"],
        }

    def __iter__(self):
        for batch1 in self.task1_dataset:
            if batch1 is not None:
                yield batch1, 1