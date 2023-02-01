# -*- coding: utf-8 -*-

import random
from collections import namedtuple
from typing import Dict
import numpy as np
import pickle
import torch
import os
from data.field import Field
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

Batch = namedtuple("Batch", ['properties', 'tweets', 'batch_size','labels']) # batch users
User = namedtuple("User", ['properties', 'tweets','tweet_max_length','label'])  # single user


class SocialbotDataset(object):

    def __init__(self,
                 dataset: str,
                 batch_size: int,
                 device: torch.device,
                 train: bool,seed=1,n_clients=5,partition="noniid-labeldir",noniid_alpha=0.3,logger=None,K=0,KFold=5,model=None,test_generalization=None):
        self.datasets = {"cresci-2015": 0, "botometer-feedback-2019": 1, "cresci-rtbust-2019": 2, "gilani-2017": 3,
                         "vendor-purchased-2019": 4, "varol-2017": 5, "Twibot-20": 6}
        self.seed = seed
        np.random.seed(seed=self.seed)
        self.cur_dataset = dataset
        base_path = os.path.dirname(os.path.dirname(__file__))
        self.base_path = base_path
        self.model = model
        self.noniid_alpha=noniid_alpha
        data_path = os.path.join(base_path, 'datas', 'NCLS-Processed', 'ZH2ENSUM')
        if self.cur_dataset =="Twibot-20":
            self.filepath = [os.path.join(base_path,"datas", "handled_data", self.raw_file_names[self.datasets[self.cur_dataset]][0]),os.path.join(base_path,"datas", "handled_data", self.raw_file_names[self.datasets[self.cur_dataset]][1])]
        else:
            self.filepath = os.path.join(base_path,"datas", "handled_data", self.raw_file_names[self.datasets[self.cur_dataset]])
        self.batch_size = batch_size
        self.train = train
        self.device = device
        self.K = K
        self.KFold = KFold
        self.n_clients = n_clients
        self.test_generalization=test_generalization
        source_field = Field(unk=True, pad=True, bos=True, eos=True, ms=True, mt=True, cls=True)
        source_vocab = os.path.join(data_path,'all_bert_voc_dic.txt')
        source_special = source_field.special
        with open(source_vocab, encoding="UTF-8") as f:
            source_words = [line.strip() for line in f]
        source_field.load_vocab(source_words, source_special)
        self.fields = source_field
        if self.test_generalization:
            datas = []
            labels = []
            b19_path = os.path.join(base_path,"datas", "handled_data", self.raw_file_names[self.datasets["botometer-feedback-2019"]])
            c19_path = os.path.join(base_path,"datas", "handled_data", self.raw_file_names[self.datasets["cresci-rtbust-2019"]])
            g17_path = os.path.join(base_path, "datas", "handled_data",self.raw_file_names[self.datasets["gilani-2017"]])
            v19_path = os.path.join(base_path, "datas", "handled_data",self.raw_file_names[self.datasets["vendor-purchased-2019"]])
            v17_path = os.path.join(base_path, "datas", "handled_data",self.raw_file_names[self.datasets["varol-2017"]])
            t20_path = [os.path.join(base_path,"datas", "handled_data", self.raw_file_names[self.datasets["Twibot-20"]][0]),os.path.join(base_path,"datas", "handled_data", self.raw_file_names[self.datasets["Twibot-20"]][1])]
            pathes = [b19_path,c19_path,g17_path,v19_path,v17_path]
            self.num_examples = 0
            for path in pathes[:5]:
                with open(path,'rb') as f:
                    data = pickle.load(f)
                users = []
                labels.append(data["labels"])
                for user_id,user_property,user_tweets,tweet_max_length,label in data['data']:
                    users.append(User(user_property,user_tweets,tweet_max_length,label))
                self.num_examples += len(users)
                datas.append(users)
            with open(t20_path[0],'rb') as f:
                data = pickle.load(f)
            users = []
            tmplabels=data["labels"]
            for user_id, user_property, user_tweets, tweet_max_length, label in data['data']:
                users.append(User(user_property, user_tweets, tweet_max_length, label))
            with open(t20_path[1],'rb') as f:
                data = pickle.load(f)
            for user_id, user_property, user_tweets, tweet_max_length, label in data['data']:
                users.append(User(user_property, user_tweets, tweet_max_length, label))
            tmplabels+=data["labels"]
            self.num_examples += len(users)
            labels.append(tmplabels)
            datas.append(users)
            self.test_batchs = list(batch(datas[self.datasets[self.cur_dataset]-1], self.batch_size))
            self.test_numbers = len(datas[self.datasets[self.cur_dataset]-1])
            self.client_batches = []
            self.num_samples = []
            self.traindata_cls_counts = {}
            for index,data in enumerate(datas):
                if index == self.datasets[self.cur_dataset]-1:
                    continue
                self.num_samples.append(len(data))
                self.client_batches.append(list(batch(data, self.batch_size)))
                unq, unq_cnt = np.unique(labels[index],return_counts=True)
                tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
                self.traindata_cls_counts[index]=tmp
            logger.info('Data statistics: %s' % str(self.traindata_cls_counts))
        else:
            if self.cur_dataset =="Twibot-20":
                user_test = []
                with open(self.filepath[1],'rb') as f:
                    test_datas = pickle.load(f)
                test_labels = test_datas['labels']
                for user_id,user_property,user_tweets,tweet_max_length,label in test_datas['data']:
                    user_test.append(User(user_property,user_tweets,tweet_max_length,label))
                self.test_batchs =list(batch(user_test,self.batch_size))
                with open(self.filepath[0],'rb') as f:
                    train_datas = pickle.load(f)
                self.train_labels = train_datas['labels']
                train_users = []
                for user_id, user_property, user_tweets, tweet_max_length, label in train_datas['data']:
                    train_users.append(User(user_property, user_tweets, tweet_max_length, label))
                user_train,_,net_dataidx_map, self.traindata_cls_counts=self.partition_data(train_users, partition, n_clients, alpha=noniid_alpha, logger=logger,only_train=True)
                self.test_numbers = len(user_test)
                self.proccess_batch_for_clients(user_train, net_dataidx_map)
            else:
                with open(self.filepath,'rb') as f:
                    datas = pickle.load(f)
                self.labels = datas['labels']
                datas = datas['data']
                users = []
                for user_id,user_property,user_tweets,tweet_max_length,label in datas:
                    users.append(User(user_property,user_tweets,tweet_max_length,label))
                self.num_examples = len(users)
                user_train,user_test,net_dataidx_map, self.traindata_cls_counts = self.partition_data(users,partition,n_clients,alpha=noniid_alpha,logger=logger)
                self.test_numbers = len(user_test)
                self.test_batchs = list(batch(user_test,self.batch_size))
                self.proccess_batch_for_clients(user_train,net_dataidx_map)
        # self.batches = list(batch(users, self.batch_size))

    @property
    def raw_file_names(self):
        return ['', 'botometer-feedback-2019.pickle', 'cresci-rtbust-2019.pickle', 'gilani-2017.pickle',
                'vendor-purchased-2019.pickle', 'varol-2017.pickle',['Twibot-20_train.pickle','Twibot-20_test.pickle']]

    def proccess_batch_for_clients(self,users_train,net_dataidx_map):
        self.client_batches = []
        self.num_samples = []
        for i in range(self.n_clients):
            users_for_i = []
            for index in net_dataidx_map[i]:
                users_for_i.append(users_train[index])
            self.num_samples.append(len(users_for_i))
            self.client_batches.append(list(batch(users_for_i,self.batch_size)))

    def load_data_i_client(self,i):
        if i =="test":
            batches = self.test_batchs
        else:
            batches = self.client_batches[i]
        for minibatch, curlen in batches:
            users_properties = []
            tweets = []
            labels = []
            for user in minibatch:
                users_properties.append(user.properties)
                length = user.tweet_max_length
                user_tweets = user.tweets
                if len(user_tweets) == 0:
                    user_tweets = [['can', "'", 't', 'find', 'this', 'user', "'", 's', 't', '##wee', '##t', '!'],
                                   ['can', "'", 't', 'find', 'this', 'user', "'", 's', 't', '##wee', '##t', '!']]
                    length = len(user_tweets[0])
                user_tweets = self.fields.process(user_tweets, self.device, length,self.model)
                tweets.append(user_tweets)
                labels.append(user.label)
                # src = self.fields.process([x.src for x in minibatch], self.device,curlen)   #minibatch里所有的padding后的句子的索引矩阵。
            # users_properties=torch.tensor(users_properties).long().to(self.device)

            yield Batch(properties=users_properties, tweets=tweets, labels=labels, batch_size=len(minibatch))  # 词的索引矩阵


    def partition_data(self,users,partition,n_clients,alpha,logger,only_train=False):
        if not only_train:
            labels = np.array(self.labels)
            skf = StratifiedKFold(n_splits=self.KFold, shuffle=True, random_state=self.seed)
            train_index,test_index = list(skf.split(list(range(len(self.labels))),self.labels))[self.K]
            y_train, y_test = labels[train_index], labels[test_index]
        else:
            y_train = np.array(self.train_labels)
        if partition =="noniid-labeldir":
            min_size = 0
            if self.cur_dataset == "Twibot-20":
                min_require_size = 100
            else:
                min_require_size = 30
            K = 2
            N = len(users)
            # np.random.seed(2020)
            net_dataidx_map = {}

            while min_size < min_require_size:
                idx_batch = [[] for _ in range(n_clients)]
                for k in range(K):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
                    # logger.info("proportions1: ", proportions)
                    # logger.info("sum pro1:", np.sum(proportions))
                    ## Balance
                    proportions = np.array([p * (len(idx_j) < N / n_clients) for p, idx_j in zip(proportions, idx_batch)])
                    # logger.info("proportions2: ", proportions)
                    proportions = proportions / proportions.sum()
                    # logger.info("proportions3: ", proportions)
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    # logger.info("proportions4: ", proportions)
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])
                    # if K == 2 and n_parties <= 10:
                    #     if np.min(proportions) < 200:
                    #         min_size = 0
                    #         break

            for j in range(n_clients):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logger)
        if not only_train:
            user_train = []
            user_test = []
            for i in train_index:
                user_train.append(users[i])
            for i in test_index:
                user_test.append(users[i])
            return (user_train, user_test,net_dataidx_map, traindata_cls_counts)
        else:
            return (users,None,net_dataidx_map,traindata_cls_counts)



    def __iter__(self):
        # while True:
            if self.train:
                random.shuffle(self.batches)
            for minibatch,curlen in self.batches:
                users_properties = []
                tweets = []
                labels = []
                for user in minibatch:
                    users_properties.append(user.properties)
                    length = user.tweet_max_length
                    user_tweets = user.tweets
                    if len(user_tweets)==0:
                        user_tweets = [['can', "'", 't', 'find', 'this', 'user', "'", 's', 't', '##wee', '##t', '!'],['can', "'", 't', 'find', 'this', 'user', "'", 's', 't', '##wee', '##t', '!']]
                        length = len(user_tweets[0])
                    user_tweets = self.fields.process(user_tweets,self.device,length)
                    tweets.append(user_tweets)
                    labels.append(user.label)
                    # src = self.fields.process([x.src for x in minibatch], self.device,curlen)   #minibatch里所有的padding后的句子的索引矩阵。
                # users_properties=torch.tensor(users_properties).long().to(self.device)

                yield Batch(properties=users_properties, tweets=tweets, labels=labels,batch_size=len(minibatch)) # 词的索引矩阵
            # if not self.train:
            #     break

    def sort(self, examples):
        seed = sorted(range(len(examples)), key=lambda idx: self.sort_key(examples[idx]))
        return sorted(examples, key=self.sort_key), seed

def batch(data, batch_size):
    minibatch, cur_len = [], 0
    for user in data:
        minibatch.append(user)
        cur_len+=1
        if cur_len> batch_size:
            yield (minibatch[:-1],cur_len-1)
            minibatch, cur_len = [user], 1
    if minibatch:
        yield (minibatch,cur_len)

def record_net_data_stats(y_train, net_dataidx_map, logger):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts