from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from data.bot_dataset import SocialbotDataset
from FLAlgorithms.utils.model_utils import load_lingual_model
import torch.nn as nn
import numpy as np

class FedEnsemble(Server):
    def __init__(self, args, model, seed,device,logger):
        super().__init__(args, model, seed)

        # Initialize data for all users
        self.dataset = SocialbotDataset(args.dataset, batch_size=args.batch_size, n_clients=args.num_users,
                                        device=device, noniid_alpha=args.noniid_alpha, train=True, logger=logger,
                                        model=model[0].extractor_model_type,test_generalization=args.test_generalization)
        # data contains: clients, groups, train_data, test_data, proxy_data
        total_users = args.num_users
        self.total_test_samples = 0
        self.slow_start = 20
        self.use_adam = args.use_adam
        self.init_ensemble_configs()
        self.init_loss_fn()
        self.device =device
        self.lingual_model = load_lingual_model()
        for p in self.lingual_model.parameters():
            p.requires_grad = False
        self.total_test_samples = self.dataset.test_numbers
        #### creating users ####
        self.users = []
        for i in range(total_users):
            self.total_train_samples+= self.dataset.num_samples[i]
            user=UserAVG(args, i, model, self.dataset,device, use_adam=True,cross_lingual_model=self.lingual_model)
            self.users.append(user)

        #### build test data loader ####
        # self.testloaderfull, self.unique_labels=aggregate_user_test_data(data, args.dataset, self.total_test_samples)
        print("Loading testing data.")
        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print("Data from {} users in total.".format(total_users))
        print("Finished creating FedAvg server.")

    def train(self, args):
        #### pretraining
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            self.selected_users, self.user_idxs=self.select_users(glob_iter, self.num_users, return_idx=True)
            self.send_parameters(mode=self.mode)
            for user_id, user in zip(self.user_idxs, self.selected_users): # allow selected users to train
                user.train(
                    glob_iter,
                    personalized=False, lr_decay=True, count_labels=True)
            self.evaluate_ensemble(selected=False)
            self.aggregate_parameters(mode=self.mode)


        self.save_results(args)
        self.save_model()
