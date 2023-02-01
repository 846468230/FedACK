from FLAlgorithms.users.userFedDistill import UserFedDistill
from FLAlgorithms.servers.serverbase import Server
from data.bot_dataset import SocialbotDataset
from FLAlgorithms.utils.model_utils import load_lingual_model
import numpy as np

class FedDistill(Server):
    def __init__(self, args, model, seed,device,logger):
        super().__init__(args, model, seed)

        # Initialize data for all users
        self.dataset = SocialbotDataset(args.dataset, batch_size=args.batch_size, n_clients=args.num_users,
                                        device=device, noniid_alpha=args.noniid_alpha, train=True,
                                        logger=logger, model=model[0].extractor_model_type,test_generalization=args.test_generalization)
        # data contains: clients, groups, train_data, test_data, proxy_data
        total_users = args.num_users
        self.lingual_model = load_lingual_model()
        self.total_test_samples = self.dataset.test_numbers
        self.slow_start = 20
        self.share_model = 'FL' in self.algorithm
        self.pretrain = 'pretrain' in self.algorithm.lower()
        self.user_logits = None
        self.init_ensemble_configs()
        self.init_loss_fn()
        self.init_ensemble_configs()
        for p in self.lingual_model.parameters():
            p.requires_grad = False
        #### creating users ####
        self.users = []
        for i in range(total_users):
            self.total_train_samples+= self.dataset.num_samples[i]
            user=UserFedDistill(
                args, i, model, self.dataset,device, self.unique_labels, use_adam=False,cross_lingual_model=self.lingual_model)
            self.users.append(user)
        print("Loading testing data.")
        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print("Data from {} users in total.".format(total_users))
        print("Finished creating FedDistill server.")

    def train(self, args):
        #### pretraining ####
        if self.pretrain:
            ## before training ##
            for iter in range(self.num_pretrain_iters):
                print("\n\n-------------Pretrain iteration number: ", iter, " -------------\n\n")
                for user in self.users:
                    user.train(iter, personalized=True, lr_decay=True)
                self.evaluate(selected=False, save=False)
            ## after training ##
            if self.share_model:
                self.aggregate_parameters(mode=self.mode)
            self.aggregate_logits(selected=False) # aggregate label-wise logit vector

        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            self.selected_users,self.user_idxs = self.select_users(glob_iter, self.num_users,return_idx=True)
            if self.share_model:
                self.send_parameters(mode=self.mode)# broadcast averaged prediction model
            self.send_logits() # send global logits if have any
            random_chosen_id = np.random.choice(self.user_idxs)
            for user_id, user in zip(self.user_idxs, self.selected_users): # allow selected users to train
                chosen = user_id == random_chosen_id
                user.train(
                    glob_iter,
                    personalized=True, lr_decay=True, count_labels=True, verbose=chosen)
            # self.evaluate()  # evaluate global model performance
            self.evaluate_personalized_model()
            if self.share_model:
                self.aggregate_parameters(mode=self.mode)
            self.aggregate_logits() # aggregate label-wise logit vector
            # self.evaluate_personalized_model()

        self.save_results(args)
        # self.save_model()

    def aggregate_logits(self, selected=True):
        user_logits = 0
        users = self.selected_users if selected else self.users
        for user in users:
            user_logits += user.logit_tracker.avg()
        self.user_logits = user_logits / len(users)

    def send_logits(self):
        if self.user_logits == None: return
        for user in self.selected_users:
            user.global_logits = self.user_logits.clone().detach()
