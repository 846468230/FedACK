from FLAlgorithms.users.userFedProx import UserFedProx
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.utils.model_utils import load_lingual_model
from data.bot_dataset import SocialbotDataset

# Implementation for FedProx Server

class FedProx(Server):
    def __init__(self, args, model, seed,device,logger):
        # dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
        #         local_epochs, num_users, K, personal_learning_rate, times):
        super().__init__(args, model,
                         seed)  # dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
        # local_epochs, num_users, times)
        self.dataset = SocialbotDataset(args.dataset, batch_size=args.batch_size, n_clients=args.num_users,
                                        device=device, noniid_alpha=args.noniid_alpha, train=True,
                                        logger=logger, model=model[0].extractor_model_type,test_generalization=args.test_generalization)
        # Initialize data for all  users
        total_users = args.num_users
        print("Users in total: {}".format(total_users))
        self.lingual_model = load_lingual_model()
        for p in self.lingual_model.parameters():
            p.requires_grad = False
        self.device = device
        for i in range(total_users):
            self.total_train_samples += self.dataset.num_samples[i]
            user = UserFedProx(args, i, model,self.dataset,device, use_adam=False,cross_lingual_model=self.lingual_model)
            self.users.append(user)

        print("Number of users / total users:", self.num_users, " / ", total_users)
        print("Finished creating FedProx server.")

    def train(self, args):
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ", glob_iter, " -------------\n\n")
            self.selected_users = self.select_users(glob_iter, self.num_users)
            self.send_parameters(mode=self.mode)
            for user in self.selected_users:  # allow selected users to train
                user.train(glob_iter)
            # self.evaluate()
            self.evaluate_personalized_model()
            self.aggregate_parameters(mode=self.mode)
        self.save_results(args)
        # self.save_model()