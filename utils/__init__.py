# -*- coding: utf-8 -*-
import os
from .saver import Saver
from .metric import calculate_bleu,calculate_rouge
def get_device(num_device=None,cpu=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{num_device}'
    import torch.cuda
    if cpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def printing_opt(opt):
    return "\n".join(["%15s | %s" % (e[0], e[1]) for e in sorted(vars(opt).items(), key=lambda x: x[0])])