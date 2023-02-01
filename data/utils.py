
# -*- coding: utf-8 -*-

from .dataset import TranslationDataset,TestDataSet
from data.field import Field

def build_dataset(opt, data_path, vocab_path, device, train=True,testdata=False,V=50):
    if not testdata:
        source_path = data_path[0]
        target_path = data_path[1]

        source_field = Field(unk=True, pad=True, bos=True, eos=True, ms=True, mt=True, cls=True)
        target_field = Field(unk=True, pad=True, bos=True, eos=True, ms=True, mt=True, cls=True)

        source_vocab, target_vocab = vocab_path
        source_special = source_field.special
        target_special = target_field.special

        with open(source_vocab, encoding="UTF-8") as f:
            source_words = [line.strip() for line in f]
        with open(target_vocab, encoding="UTF-8") as f:
            target_words = [line.strip() for line in f]

        source_field.load_vocab(source_words, source_special)
        target_field.load_vocab(target_words, target_special)

        data = TranslationDataset(source_path, target_path, opt.batch_size, device, train,
                               {'src': source_field, 'tgt': target_field}) # MT

        return data
    else:
        source_field = Field(unk=True, pad=True, bos=True, eos=True, ms=True, mt=True, cls=True)
        target_field = Field(unk=True, pad=True, bos=True, eos=True, ms=True, mt=True, cls=True)
        source_special = source_field.special
        target_special = target_field.special

        source_words = [ str(i) for i in range(1,V) ]
        target_words = source_words[:]

        source_field.load_vocab(source_words, source_special)
        target_field.load_vocab(target_words, target_special)
        data = TestDataSet("", "", opt.batch_size, device, train,
                                  {'src': source_field, 'tgt': target_field},V=V,num_examples=opt.test_data_num)
        return data


