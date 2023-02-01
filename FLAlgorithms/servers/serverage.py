from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from data.bot_dataset import SocialbotDataset
from FLAlgorithms.utils.model_utils import load_lingual_model
import numpy as np
# Implementation for FedAvg Server
import time


class FedAvg(Server):
    def __init__(self, args,  model, seed,device,logger):
        super().__init__(args, model, seed)

        # Initialize data for all  users
        self.dataset = SocialbotDataset(args.dataset, batch_size=args.batch_size,n_clients=args.num_users,device=device, noniid_alpha=args.noniid_alpha,train=True,
                                        logger=logger,model=model[0].extractor_model_type,test_generalization=args.test_generalization)
        # data contains: clients, groups, train_data, test_data, proxy_data
        total_users = args.num_users
        self.lingual_model = load_lingual_model()
        for p in self.lingual_model.parameters():
            p.requires_grad = False
        self.use_adam = 'adam' in self.algorithm.lower()
        print("Users in total: {}".format(total_users))
        self.device = device
        for i in range(total_users):
            self.total_train_samples+= self.dataset.num_samples[i]
            user = UserAVG(args, i, model, self.dataset,device, use_adam=True,cross_lingual_model=self.lingual_model)
            self.users.append(user)

        print("Number of users / total users:", args.num_users, " / ", total_users)
        print("Finished creating FedAvg server.")

    def train(self, args):
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ", glob_iter, " -------------\n\n")
            self.selected_users = self.select_users(glob_iter, self.num_users)
            self.send_parameters(mode=self.mode)
            self.timestamp = time.time()  # log user-training start time
            for user in self.selected_users:  # allow selected users to train
                user.train(glob_iter, personalized=self.personalized)  # * user.train_samples
            curr_timestamp = time.time()  # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)
            # self.evaluate()
            self.evaluate_personalized_model()
            # Evaluate selected user
            # if self.personalized:
            #     # Evaluate personal model on user for each iteration
            #     print("Evaluate personal model\n")
            #     self.evaluate_personalized_model()

            self.timestamp = time.time()  # log server-agg start time
            self.aggregate_parameters(self.mode)
            curr_timestamp = time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)
        self.save_results(args)
        # self.save_model()