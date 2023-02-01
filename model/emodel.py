# -*- coding: utf-8 -*-
from typing import Dict

import torch
import torch.nn as nn
import numpy as np
from .embeddings import Embedding
from .transformer import Decoder, Encoder
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, hidden_size: int, tgt_vocab_size: int):
        self.vocab_size = tgt_vocab_size
        super(Generator, self).__init__()
        self.linear_hidden = nn.Linear(hidden_size, tgt_vocab_size)
        self.lsm = nn.LogSoftmax(dim=-1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_hidden.weight)

    def forward(self, dec_out):
        score = self.linear_hidden(dec_out)
        lsm_score = self.lsm(score)
        return lsm_score

class Discriminator(nn.Module):
    def __init__(self,opt):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.hidden_size, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        self.model.apply(init_weights)
        self.dis_loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.adam_lr, betas=(opt.b1, opt.b2))
    def forward(self, rep):
        # Concatenate source_rep and target_rep
        validity = self.model(rep)
        return validity


class Mapper(nn.Module):
    def __init__(self,hidden_size:int,opt):
        super(Mapper, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        self.model = nn.Sequential(
            *block(hidden_size, hidden_size*2),
            *block(hidden_size*2, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.model.apply(init_weights)
        self.loss_f = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.adam_lr, betas=(opt.b1, opt.b2))



    def forward(self, enc_out):
        batch_size = enc_out.size(0)
        dim = enc_out.size(-1)
        enc_out = enc_out.view(-1,dim)
        out = self.model(enc_out)
        out = out.view(batch_size,-1,dim)
        return out


class EModel(nn.Module):

    def __init__(self, encoder: Encoder,
                 decoder: Decoder,
                 generator: Generator,
                 discriminator:Discriminator,
                 mapper: Mapper,
                 model_opt,
                 device,
                 ):
        super(EModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.discriminator = discriminator
        self.mapper = mapper
        self.device =device

    def sample_z(self, mu, logvar):
        eps = torch.rand_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def sample_z1(self, mu, logvar):
        epsilon = logvar.new_empty(logvar.size()).normal_()
        std = torch.exp(0.5 * logvar)
        z = mu + std * epsilon
        return z

    def gaussian_kld(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                               - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                               - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
        return kld

    def softplus(self, x):
        return torch.log(1.0 + torch.exp(x))

    def forward_to_embedding(self,source,mapper=False):
        source_pad = source.eq(self.encoder.embedding.word_padding_idx)
        source_enc_out = self.encoder(source, source_pad)
        if mapper:
            source_enc_out = self.mapper(source_enc_out)
        return source_enc_out,source_pad

    def forward(self, source, ori_target, is_training=False,training_dis=2):    #输入是 原句子索引矩阵，目标句子索引矩阵
        # target = ori_target[:, :-1]  # shift left
        target = ori_target[:,:]
        source_pad = source.eq(self.encoder.embedding.word_padding_idx)     #根据填充id找到填充的位置置1
        target_pad = target.eq(self.decoder.embedding.word_padding_idx)
        source_enc_out = self.encoder(source, source_pad)
        if is_training: #train dis
            target_enc_out = self.encoder(target, target_pad)
            source_mapper_to_target = self.mapper(source_enc_out)
            target_mapper_to_source = self.mapper(target_enc_out)
            source_mapper_to_target_rep = torch.mean(source_mapper_to_target, dim=1, keepdim=False)
            target_mapper_to_source_rep = torch.mean(target_mapper_to_source, dim=1, keepdim=False)
            if training_dis==0:
                source_rep = torch.mean(source_enc_out, dim=1, keepdim=False)
                target_rep = torch.mean(target_enc_out, dim=1, keepdim=False)

                # Loss for real
                source_valid = Variable(torch.FloatTensor(source_rep.size(0), 1).fill_(1.0), requires_grad=False).to(self.device)
                validity_real_source = self.discriminator(source_rep)
                source_rep_real_loss = self.discriminator.dis_loss(validity_real_source, source_valid)
                target_valid = Variable(torch.FloatTensor(target_rep.size(0), 1).fill_(1.0), requires_grad=False).to(self.device)
                validity_real_target = self.discriminator(target_rep)
                target_rep_real_loss = self.discriminator.dis_loss(validity_real_target, target_valid)

                # Loss for fake
                target_to_source_valid = Variable(torch.FloatTensor(target_mapper_to_source_rep.size(0), 1).fill_(0.0), requires_grad=False).to(self.device)
                validity_fake_source = self.discriminator(target_mapper_to_source_rep)
                source_rep_fake_loss = self.discriminator.dis_loss(validity_fake_source, target_to_source_valid)
                source_to_target_valid = Variable(torch.FloatTensor(source_mapper_to_target_rep.size(0), 1).fill_(0.0), requires_grad=False).to(self.device)
                validity_fake_target = self.discriminator(source_mapper_to_target_rep)
                target_rep_fake_loss = self.discriminator.dis_loss(validity_fake_target, source_to_target_valid)
                dis_loss = (source_rep_real_loss+target_rep_real_loss+source_rep_fake_loss+target_rep_fake_loss) / 4
                return 0,0,0,0,dis_loss
            elif training_dis==1: #train gen
                #gen label
                gen_source_valid = Variable(torch.FloatTensor(target_mapper_to_source_rep.size(0), 1).fill_(1.0),requires_grad=False).to(self.device)
                gen_target_valid = Variable(torch.FloatTensor(source_mapper_to_target_rep.size(0), 1).fill_(1.0), requires_grad=False).to(self.device)
                gen_source_loss = self.mapper.loss_f(self.discriminator(target_mapper_to_source_rep),gen_source_valid)
                gen_target_loss = self.mapper.loss_f(self.discriminator(source_mapper_to_target_rep),gen_target_valid)
                gen_loss = (gen_source_loss + gen_target_loss) / 2
                return 0,0,0,0,gen_loss
            else:  # train transformer
                source_to_target_outputs, _ = self.decoder(target, source_enc_out, source_pad, target_pad)
                target_to_source_outputs, _ = self.decoder(source, target_enc_out, target_pad, source_pad)
                target_mapper_to_source_outputs,_ = self.decoder(target,target_mapper_to_source,source_pad,target_pad)
                source_mapper_to_target_outputs,_ = self.decoder(source,source_mapper_to_target,target_pad,source_pad)
                return self.generator(source_to_target_outputs),self.generator(target_to_source_outputs),self.generator(target_mapper_to_source_outputs),self.generator(source_mapper_to_target_outputs),0
        else:
            source_to_target_outputs, _ = self.decoder(target, source_enc_out, source_pad, target_pad)
            return self.generator(source_to_target_outputs),0,0,0,0

    def test_forward(self, source, ori_target):
        target = ori_target[:, :]
        source_pad = source.eq(self.encoder.embedding.word_padding_idx)  # 根据填充id找到填充的位置置1
        target_pad = target.eq(self.decoder.embedding.word_padding_idx)
        source_enc_out = self.encoder(source, source_pad)
        target_enc_out = self.encoder(target, target_pad)
        source_mapper_to_target = self.mapper(source_enc_out)
        target_mapper_to_source = self.mapper(target_enc_out)
        source_to_target_outputs, _ = self.decoder(target, source_enc_out, source_pad, target_pad)
        target_to_source_outputs, _ = self.decoder(source, target_enc_out, target_pad, source_pad)
        target_mapper_to_source_outputs, _ = self.decoder(target, target_mapper_to_source, source_pad, target_pad)
        source_mapper_to_target_outputs, _ = self.decoder(source, source_mapper_to_target, target_pad, source_pad)
        return self.generator(source_to_target_outputs), self.generator(target_to_source_outputs), self.generator(
            target_mapper_to_source_outputs), self.generator(source_mapper_to_target_outputs), 0
    @classmethod
    def load_model(cls, model_opt,
                   device,
                   pad_ids: Dict[str, int],
                   vocab_sizes: Dict[str, int],
                   checkpoint=None):
        source_embedding = Embedding(embedding_dim=model_opt.hidden_size,
                                     dropout=model_opt.dropout,
                                     padding_idx=pad_ids["src"],
                                     vocab_size=vocab_sizes["src"])
        # MT
        if model_opt.share_source_target_embedding:
            target_embedding = source_embedding
        else:
            target_embedding = Embedding(embedding_dim=model_opt.hidden_size,
                                         dropout=model_opt.dropout,
                                         padding_idx=pad_ids["tgt"],
                                         vocab_size=vocab_sizes["tgt"])
        encoder = Encoder(model_opt.layers,
                          model_opt.heads,
                          model_opt.hidden_size,
                          model_opt.dropout,
                          model_opt.ff_size,
                          source_embedding)

        decoder = Decoder(model_opt.layers,
                          model_opt.heads,
                          model_opt.hidden_size,
                          model_opt.dropout,
                          model_opt.ff_size,
                          target_embedding)

        generator = Generator(model_opt.hidden_size, vocab_sizes["tgt"])
        discriminator = Discriminator(model_opt)
        mapper = Mapper(model_opt.hidden_size,model_opt)
        model = cls(encoder, decoder, generator,discriminator,mapper,model_opt,device)
        if checkpoint is None and model_opt.train_from:
            checkpoint = torch.load(model_opt.train_from, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint["model"])
        elif checkpoint is not None:
            model.load_state_dict(checkpoint)
        return model
