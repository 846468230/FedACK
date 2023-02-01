# -*- coding: utf-8 -*-
from typing import List

import torch

EOS_TOKEN = "<eos>"
BOS_TOKEN = "<bos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

MS_TOKEN = "<MS>"
MT_TOKEN = "<MT>"
CLS_TOKEN = "<CLS>"

class Field(object):
    def __init__(self, bos: bool, eos: bool, pad: bool, unk: bool, ms: bool, mt: bool, cls: bool):
        self.bos_token = BOS_TOKEN if bos else None
        self.eos_token = EOS_TOKEN if eos else None
        self.unk_token = UNK_TOKEN if unk else None
        self.pad_token = PAD_TOKEN if pad else None

        self.ms_token = MS_TOKEN if ms else None
        self.mt_token = MT_TOKEN if mt else None
        self.cls_token = CLS_TOKEN if cls else None

        self.vocab = None

    def load_vocab(self, words: List[str], specials: List[str]):
        self.vocab = Vocab(words, specials)

    def process(self, batch, device,length,model=None):
        cut_length=48
        # max_len = max(len(x) for x in batch)
        max_len = min(length,cut_length)
        if model.lower() == "cnn":
            max_len=cut_length
        padded, length = [], []

        for x in batch:
            x = x[:cut_length]
            bos = [self.bos_token] if self.bos_token else []
            eos = [self.eos_token] if self.eos_token else []
            ms = [self.ms_token] if self.ms_token else []
            mt = [self.mt_token] if self.mt_token else []
            cls = [self.cls_token] if self.cls_token else []
            pad = [self.pad_token] * (max_len - len(x))
            padded.append(bos + x + eos + pad)

            length.append(len(x) + len(bos) + len(eos))
#             length.append(len(x) + len(bos) + len(eos) + len(ms) + len(mt) + len(cls))

        padded = torch.tensor([self.encode(ex) for ex in padded])   # padded 是一堆填充了词首和词尾以及padding的句子列表。返回了这些token的索引矩阵create dictionary 返回的batch 索引矩阵。

        return padded.long().to(device)

    def encode(self, tokens):          # encode the tokens
        ids = []
        for tok in tokens:
            if tok in self.vocab.stoi:
                ids.append(self.vocab.stoi[tok])
            else:
                ids.append(self.unk_id)
        return ids

    def decode(self, ids):            # decode the tokens
        tokens = []
        for tok in ids:
            tok = self.vocab.itos[tok]
            if tok == self.eos_token:
                break
            if tok == self.bos_token:
                continue
            tokens.append(tok)
        # if flag:
        #     return " ".join(tokens).replace("@@ ", "").replace("@@", "")
        # else:
        tokens = [ str(tok)for tok in tokens]
        return " ".join(tokens).replace("@@ ", "").replace("@@", "")

    @property
    def special(self):
        return [tok for tok in [self.unk_token, self.pad_token, self.bos_token, self.eos_token, self.ms_token, self.mt_token, self.cls_token] if tok is not None]

    @property
    def pad_id(self):
        return self.vocab.stoi[self.pad_token]

    @property
    def eos_id(self):
        return self.vocab.stoi[self.eos_token]

    @property
    def bos_id(self):
        return self.vocab.stoi[self.bos_token]

    @property
    def unk_id(self):
        return self.vocab.stoi[self.unk_token]


class Vocab(object):
    def __init__(self, words: List[str], specials: List[str]):
        self.itos = specials + words
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)