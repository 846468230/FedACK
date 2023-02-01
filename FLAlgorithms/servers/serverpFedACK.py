from FLAlgorithms.users.userpFedACK import UserpFedACK
from FLAlgorithms.servers import Server
from FLAlgorithms.utils.model_utils import read_data, read_user_data, aggregate_user_data, create_generative_model,load_lingual_model
from data.bot_dataset import SocialbotDataset
import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import time
MIN_SAMPLES_PER_LABEL=1

class FedACK(Server):
    def __init__(self, args, model, seed,device,logger):
        super().__init__(args, model, seed)

        # Initialize data for all users
        self.dataset = SocialbotDataset(args.dataset, batch_size=args.batch_size,n_clients=args.num_users, device=device, noniid_alpha=args.noniid_alpha,train=True,logger=logger,model=model[0].extractor_model_type,test_generalization=args.test_generalization)
        total_users = args.num_users
        self.total_test_samples = 0
        self.local = 'local' in self.algorithm.lower()
        self.visualize = args.visualize_boundary
        self.use_adam = args.use_adam #'adam' in self.algorithm.lower()
        self.early_stop = 20  # stop using generated samples after 20 local epochs
        self.generative_model = create_generative_model(args.dataset, args.algorithm, args.embedding,args.visualize_boundary)
        self.model_epochs = args.local_epochs
        if not args.train:
            print('number of generator parameteres: [{}]'.format(self.generative_model.get_number_of_parameters()))
            print('number of model parameteres: [{}]'.format(self.model.get_number_of_parameters()))
        self.latent_layer_idx = self.generative_model.latent_layer_idx
        self.init_ensemble_configs()
        print("latent_layer_idx: {}".format(self.latent_layer_idx))
        print("label embedding {}".format(self.generative_model.embedding))
        print("ensemeble learning rate: {}".format(self.ensemble_lr))
        print("ensemeble alpha = {}, beta = {}, eta = {}".format(self.ensemble_alpha, self.ensemble_beta, self.ensemble_eta))
        print("generator alpha = {}, beta = {}".format(self.generative_alpha, self.generative_beta))
        self.init_loss_fn()
        self.device = device
        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)

        #### creating users ####
        self.lingual_model = load_lingual_model()
        for p in self.lingual_model.parameters():
            p.requires_grad = False
        self.users = []
        self.total_test_samples = self.dataset.test_numbers
        for i in range(total_users):
            self.total_train_samples+= self.dataset.num_samples[i]

            user=UserpFedACK(
                args, i, model, self.generative_model,self.generative_optimizer,
                self.dataset,
                device,
                self.latent_layer_idx, #label_info,
                use_adam=self.use_adam,cross_lingual_model=self.lingual_model,ensemble_epochs=self.ensemble_epochs,n_teacher_iters=self.n_teacher_iters,unique_labels=self.unique_labels,
                ensemble_alpha=self.ensemble_alpha,ensemble_beta=self.ensemble_beta,ensemble_eta=self.ensemble_eta,ensemble_lr=self.ensemble_lr,weight_decay=self.weight_decay)
            self.users.append(user)
        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print("Data from {} users in total.".format(total_users))
        print(f"Finished creating {args.algorithm} server.")

    def train(self, args):
        #### pretraining
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            self.selected_users=self.select_users(glob_iter, self.num_users, return_idx=False)
            if not self.local:
                self.send_parameters(mode=self.mode)# broadcast averaged prediction model
            # self.evaluate()
            chosen_verbose_user = np.random.randint(0, len(self.users))
            self.timestamp = time.time() # log user-training start time
            for user in self.selected_users: # allow selected users to train
                user_id = user.id
                verbose= user_id == chosen_verbose_user
                # perform regularization using generated samples after the first communication round
                user.train(
                    glob_iter,
                    personalized=self.personalized,
                    early_stop=self.early_stop,
                    verbose=verbose and glob_iter > 0,
                    regularization= glob_iter > 0 )
            curr_timestamp = time.time() # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)
            if self.personalized and not self.visualize:
                self.evaluate_personalized_model()      # need to redefine

            self.timestamp = time.time() # log server-agg start time
            self.train_generator(
                self.batch_size,
                epoches=self.ensemble_epochs // self.n_teacher_iters,
                latent_layer_idx=self.latent_layer_idx,
                verbose=True
            )
            self.aggregate_parameters(self.mode)
            curr_timestamp=time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)

        self.save_results(args)
        self.save_model()

    def train_generator(self, batch_size, epoches=1, latent_layer_idx=-1, verbose=False):
        """
        Learn a generator that find a consensus latent representation z, given a label 'y'.
        :param batch_size:
        :param epoches:
        :param latent_layer_idx: if set to -1 (-2), get latent representation of the last (or 2nd to last) layer.
        :param verbose: print loss information.
        :return: Do not return anything.
        """
        self.label_weights, self.qualified_labels = self.get_label_weights()
        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0
        self.generative_model.to_device(self.device)
        def update_generator_(n_iters, student_model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS):
            self.generative_model.train()
            student_model.eval()
            student_model.to_device()
            for i in range(n_iters):
                self.generative_optimizer.zero_grad()
                label_ration = self.get_label_ratio()
                y=np.random.choice(self.unique_labels, batch_size,p=label_ration.ravel())
                y_input=torch.LongTensor(y).to(self.device)
                ## feed to generator
                gen_result=self.generative_model(y_input, latent_layer_idx=latent_layer_idx, verbose=True)
                gen_output, eps=gen_result['output'], gen_result['eps']
                ##### get losses ####
                diversity_loss=self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs

                ######### get teacher loss ############
                teacher_loss=0
                teacher_logit=0
                student_loss=0
                for user_idx, user in enumerate(self.selected_users):
                    weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
                    expand_weight=np.tile(weight, (1, self.unique_labels))
                    user_result_given_gen=user.model.forward_to_classify(gen_output, logit=True)
                    user_output_logp_=F.log_softmax(user_result_given_gen['logit'], dim=1)
                    teacher_loss_=torch.mean( \
                        self.generative_model.crossentropy_loss(user_output_logp_, y_input) * \
                        torch.tensor(weight, dtype=torch.float32).to(self.device))
                    teacher_loss+=teacher_loss_
                    teacher_logit=user_result_given_gen['logit']
                    student_output = student_model.forward_to_classify(gen_output, logit=True)
                    student_loss+= torch.mean(F.kl_div(F.log_softmax(student_output['logit'], dim=1),
                                            F.softmax(teacher_logit, dim=1)) * torch.tensor(expand_weight, dtype=torch.float32).to(self.device))
                ######### get student loss ############
                if self.ensemble_beta > 0:
                    loss=self.ensemble_alpha * teacher_loss - self.ensemble_beta * student_loss + self.ensemble_eta * diversity_loss
                else:
                    loss=self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss
                loss.backward()
                self.generative_optimizer.step()
                TEACHER_LOSS += self.ensemble_alpha * teacher_loss#
                STUDENT_LOSS += self.ensemble_beta * student_loss#
                DIVERSITY_LOSS += self.ensemble_eta * diversity_loss#
            return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS
        for user_idx, user in enumerate(self.selected_users):
            user.model.to_device()
            user.model.eval()
        for i in range(epoches):
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS=update_generator_(
                self.n_teacher_iters, self.model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
        for user_idx, user in enumerate(self.selected_users):
            user.model.move_from_device()
        self.model.move_from_device()
        TEACHER_LOSS = TEACHER_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * epoches)
        STUDENT_LOSS = STUDENT_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * epoches)
        DIVERSITY_LOSS = DIVERSITY_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * epoches)
        info="Global Generator on Server: Teacher Loss= {:.4f}, Student Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
            format(TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
        if verbose:
            print(info)
        self.generative_lr_scheduler.step()


    def train_model(self, batch_size,epoches):
        self.label_weights, self.qualified_labels = self.get_label_weights()
        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0
        self.generative_model.to_device(self.device)
        def update_model_(n_iters, student_model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS):
            self.generative_model.eval()
            student_model.train()
            student_model.to_device()
            for i in range(n_iters):
                self.optimizer.zero_grad()
                label_ration = self.get_label_ratio()
                y=np.random.choice(self.unique_labels, batch_size,p=label_ration.ravel())
                y_input=torch.LongTensor(y).to(self.device)
                gen_result=self.generative_model(y_input, latent_layer_idx= -1, verbose=True)
                gen_output, eps=gen_result['output'], gen_result['eps']
                ##### get losses ####
                diversity_loss=-self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs

                ######### get teacher loss ############
                teacher_loss=0
                teacher_logit=0
                student_loss=0
                for user_idx, user in enumerate(self.selected_users):
                    weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
                    expand_weight=np.tile(weight, (1, self.unique_labels))
                    user_result_given_gen=user.model.forward_to_classify(gen_output, logit=True)
                    user_output_logp_=F.log_softmax(user_result_given_gen['logit'], dim=1)
                    teacher_loss_=torch.mean( \
                        self.generative_model.crossentropy_loss(user_output_logp_, y_input) * \
                        torch.tensor(weight, dtype=torch.float32).to(self.device))
                    teacher_loss+=-teacher_loss_
                    teacher_logit=user_result_given_gen['logit']
                    student_output = student_model.forward_to_classify(gen_output, logit=True)
                    student_loss+= -torch.mean(F.kl_div(F.log_softmax(student_output['logit'], dim=1),
                                            F.softmax(teacher_logit, dim=1)) * torch.tensor(expand_weight, dtype=torch.float32).to(self.device))
                ######### get student loss ############
                if self.ensemble_beta > 0:
                    loss=self.ensemble_alpha * teacher_loss - self.ensemble_beta * student_loss + self.ensemble_eta * diversity_loss
                else:
                    loss=self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss
                loss.backward()
                self.optimizer.step()
                TEACHER_LOSS += self.ensemble_alpha * teacher_loss#
                STUDENT_LOSS += self.ensemble_beta * student_loss#
                DIVERSITY_LOSS += self.ensemble_eta * diversity_loss#
            return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS
        for user_idx, user in enumerate(self.selected_users):
            user.model.to_device()
            user.model.eval()
        for i in range(epoches):
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS=update_model_(
                self.n_teacher_iters, self.model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
        for user_idx, user in enumerate(self.selected_users):
            user.model.move_from_device()
        self.model.move_from_device()
        TEACHER_LOSS = TEACHER_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * epoches)
        STUDENT_LOSS = STUDENT_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * epoches)
        DIVERSITY_LOSS = DIVERSITY_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * epoches)
        info="Global Model on Server: Teacher Loss= {:.4f}, Student Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
            format(TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
        print(info)
        self.lr_scheduler.step()

    def add_parameters(self, user, ratio, mode='all'):
        if mode=='partial':
            for server_param, user_param in zip(self.model.get_shared_parameters(), user.model.get_shared_parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio
        elif mode=='all':
            for server_param, user_param in zip(self.model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self,mode='all'):
        assert (self.selected_users is not None and len(self.selected_users) > 0)
        if mode == 'partial':
            for param in self.model.get_shared_parameters():
                param.data = torch.zeros_like(param.data)
        elif mode=='all':
            for param in self.model.parameters():
                param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train,mode=mode)

    def send_parameters(self, mode='all', beta=1, selected=False):
        users = self.users
        if selected:
            assert (self.selected_users is not None and len(self.selected_users) > 0)
            users = self.selected_users
        for user in users:
            if mode == 'all': # share only subset of parameters
                user.set_parameters(self.model,beta=beta)
            else: # share all parameters
                user.set_shared_parameters(self.model)

    def get_label_ratio(self):
        label_counts = []
        for label in range(self.unique_labels):
            labels = 0
            for user in self.selected_users:
                labels+=user.label_counts[label]
            label_counts.append(labels)
        label_ratios = np.array(label_counts) / np.sum(label_counts)
        return label_ratios

    def get_label_weights(self):
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for user in self.selected_users:
                weights.append(user.label_counts[label])
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            label_weights.append( np.array(weights) / np.sum(weights) )
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        return label_weights, qualified_labels
