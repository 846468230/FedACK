import torch
import torch.nn.functional as F
import numpy as np
from FLAlgorithms.users import User
from FLAlgorithms.optimizers import pFedIBOptimizer
import copy
from itertools import chain

class UserpFedACK(User):
    def __init__(self,
                 args, id, model, generative_model,generative_optimizer,
                 dataset,device, latent_layer_idx, #label_info,
                 use_adam=False,cross_lingual_model=None,ensemble_epochs=None,
                 n_teacher_iters=5,unique_labels=2,ensemble_alpha=1,ensemble_beta=1,ensemble_eta=1,ensemble_lr=0.01,weight_decay=1e-2):
        super().__init__(args, id, model, dataset, use_adam=use_adam)
        self.gen_batch_size = args.gen_batch_size
        self.generative_model = generative_model
        self.generative_local = copy.deepcopy(generative_model)
        self.student_model = copy.deepcopy(self.model)
        self.student_model.extractor = self.model.extractor
        self.cross_lingual_model = cross_lingual_model
        self.latent_layer_idx = latent_layer_idx
        # self.label_info=label_info
        self.device = device
        self.teacher_trained = False
        self.train_samples = self.datas.num_samples[self.id]
        self.ensemble_epochs =ensemble_epochs
        self.n_teacher_iters = n_teacher_iters
        self.unique_labels = unique_labels
        self.generative_local_optimizer = torch.optim.Adam(
            params=self.generative_local.parameters(),
            lr=ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_local_optimizer, gamma=0.98)
        self.use_adam = use_adam
        if use_adam:
            # self.d1_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()),lr=self.learning_rate, betas=(0.9, 0.999),eps=1e-08, weight_decay=1e-2, amsgrad=False)
            self.d1_optimizer = torch.optim.Adam(params=self.model.MLP_layers.parameters(),lr=self.learning_rate, betas=(0.9, 0.999),eps=1e-08, weight_decay=1e-2, amsgrad=False)
            self.d2_optimizer = torch.optim.Adam(params=self.student_model.MLP_layers.parameters(),lr=self.learning_rate, betas=(0.9, 0.999),eps=1e-08, weight_decay=1e-2, amsgrad=False)
            self.both_12_optimizer = torch.optim.Adam(params=chain(self.model.MLP_layers.parameters(),self.student_model.MLP_layers.parameters(),),
                                                 lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                                 weight_decay=1e-2, amsgrad=False)
            self.extractor_optimizer = torch.optim.Adam(self.model.extractor.parameters(), lr=self.learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-2, amsgrad=False)
        else:
            self.d1_optimizer = pFedIBOptimizer(params=self.model.MLP_layers.parameters(), lr=self.learning_rate)
            self.d2_optimizer = pFedIBOptimizer(params=self.student_model.MLP_layers.parameters(),lr=self.learning_rate)
            self.both_12_optimizer = pFedIBOptimizer(params=chain(self.model.MLP_layers.parameters(),self.student_model.MLP_layers.parameters(),),lr=self.learning_rate),
            self.extractor_optimizer = pFedIBOptimizer(params=self.model.extractor.parameters(),lr=self.learning_rate)
        self.d1_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.d1_optimizer, gamma=0.99)
        self.d2_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.d2_optimizer, gamma=0.99)
        self.both_12_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.both_12_optimizer, gamma=0.99)
        self.extractor_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.extractor_optimizer, gamma=0.99)
        # self.optimizer = self.reinit_optimizer(self.model,self.use_adam)
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        self.ensemble_alpha = ensemble_alpha
        self.ensemble_beta = ensemble_beta
        self.ensemble_eta = ensemble_eta
        self.eta = 0.5
        self.temperature = 0.5
        self.mu = 0.5
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels):
        for label in labels:
            self.label_counts[int(label)] += 1

    def set_parameters(self, model,beta=1):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            if beta == 1:
                old_param.data = new_param.data.clone()
            else:
                old_param.data = beta * new_param.data.clone() + (1 - beta)  * old_param.data.clone()

    def set_shared_parameters(self, model,mode="partial"):
        # only copy shared extractor's parameters
        if mode=="partial":
            for old_param, new_param in zip(self.model.get_shared_parameters(), model.get_shared_parameters()):
                old_param.data = new_param.data.clone()

    def test_personalized_student_model(self):
        self.student_model.to_device()
        self.student_model.eval()
        test_acc = 0
        loss = 0
        for i, batch in enumerate(self.datas.load_data_i_client("test")):
            with torch.no_grad():
                output = self.student_model(batch,self.cross_lingual_model)['output']
                labels = torch.tensor(batch.labels).to(self.device)
                loss += self.ce_loss(output, labels).item()
                test_acc += (torch.sum(torch.argmax(output, dim=1) == labels)).item()
        self.student_model.move_from_device()
        return test_acc, self.datas.test_numbers, loss

    def test_personalized_model(self):
        self.model.to_device()
        self.model.eval()
        test_acc = 0
        loss = 0
        for i, batch in enumerate(self.datas.load_data_i_client("test")):
            with torch.no_grad():
                output = self.model(batch,self.cross_lingual_model)['output']
                labels = torch.tensor(batch.labels).to(self.device)
                loss += self.ce_loss(output, labels).item()
                test_acc += (torch.sum(torch.argmax(output, dim=1) == labels)).item()
        self.model.move_from_device()
        return test_acc, self.datas.test_numbers, loss

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label:1 for label in range(self.unique_labels)}

    def train(self, glob_iter, personalized=False, early_stop=100, regularization=True, verbose=False):
        self.model.to_device()
        self.model.train()
        self.generative_model.to_device(self.device)
        self.generative_local.to_device(self.device)
        self.student_model.to_device()
        self.clean_up_counts()
        self.updated_labels = False
        for param in self.model.extractor.parameters():
            param.requires_grad = False
        self.global_extractor = copy.deepcopy(self.model.extractor)
        for param in self.global_extractor.parameters():
            param.requires_grad = False
        for epoch in range(self.local_epochs):
        # for epoch in range(self.train_teacher_epochs):
                # for i in range(len(self.iter_trainloader)):
            total_loss=0
            for i, batch in enumerate(self.datas.load_data_i_client(self.id)):
                eta = self.eta
                self.both_12_optimizer.zero_grad()
                labels = torch.tensor(batch.labels).to(self.device)
                if self.updated_labels == False:
                    self.update_label_counts(labels)
                d1_model_result = self.model(batch, self.cross_lingual_model, logit=True, e_reps=True)
                d1_predictive_loss = self.loss(F.log_softmax(d1_model_result['logit'],dim=1), labels)
                d2_model_result = self.student_model.forward_to_classify(d1_model_result['feature'],logit=True)
                d2_predictive_loss = self.loss(F.log_softmax(d2_model_result['logit'], dim=1), labels)
                diversity_loss = self.ensemble_loss(F.log_softmax(d2_model_result['logit'],dim=1),F.softmax(d1_model_result['logit'],dim=1))
                loss = d1_predictive_loss + d2_predictive_loss - eta * diversity_loss
                if regularization and epoch < early_stop and labels.size(0) > 1:
                    generative_beta = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_beta)
                    generative_alpha = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_alpha)
                    gen_output = self.generative_model(labels, latent_layer_idx=self.latent_layer_idx)['output']
                    logit_given_gen = self.model.forward_to_classify(gen_output, logit=True)['logit']
                    target_p = F.softmax(logit_given_gen, dim=1).clone().detach()
                    user_latent_loss_d1 = generative_beta * self.ensemble_loss(F.log_softmax(d1_model_result['logit'],dim=1), target_p)
                    logit_given_gen = self.student_model.forward_to_classify(gen_output, logit=True)['logit']
                    target_p = F.softmax(logit_given_gen, dim=1).clone().detach()
                    user_latent_loss_d2 = generative_beta * self.ensemble_loss(F.log_softmax(d2_model_result['logit'], dim=1), target_p)

                    sampled_y = np.random.choice(self.unique_labels, self.gen_batch_size)
                    sampled_y = torch.tensor(sampled_y).to(self.device)
                    gen_result = self.generative_model(sampled_y, latent_layer_idx=self.latent_layer_idx)
                    gen_output = gen_result['output']  # latent representation when latent = True, x otherwise
                    user_output_logp = self.model.forward_to_classify(gen_output, logit=False)['output']
                    teacher_loss_d1 = generative_alpha * torch.mean(
                        self.generative_model.crossentropy_loss(user_output_logp, sampled_y)
                    )
                    user_output_logp = self.student_model.forward_to_classify(gen_output, logit=False)['output']
                    teacher_loss_d2 = generative_alpha * torch.mean(
                        self.generative_model.crossentropy_loss(user_output_logp, sampled_y)
                    )

                    gen_local_result = self.generative_local(sampled_y, latent_layer_idx=self.latent_layer_idx)
                    gen_local_output = gen_local_result['output']  # latent representation when latent = True, x otherwise
                    user_output_logp = self.model.forward_to_classify(gen_local_output, logit=True)['logit']
                    user_output_logp_local = self.student_model.forward_to_classify(gen_local_output, logit=True)['logit']
                    class_local_div_loss = self.eta * self.ensemble_loss(F.log_softmax(user_output_logp_local,dim=1), F.softmax(user_output_logp,dim=1))

                    gen_ratio = self.gen_batch_size / self.batch_size
                    loss += user_latent_loss_d1+user_latent_loss_d2  + gen_ratio * (teacher_loss_d1+teacher_loss_d2) + class_local_div_loss
                total_loss += loss.item()
                loss.backward()
                self.both_12_optimizer.step()  # self.local_model)
            self.updated_labels = True
            print(f"User {self.id} training discriminators on epoch {epoch} loss is : {total_loss / (i + 1):.4f}")
        self.both_12_lr_scheduler.step(glob_iter)
        for param in self.model.extractor.parameters():
            param.requires_grad = True
        for param in self.student_model.MLP_layers.parameters():
            param.requires_grad = False
        for param in self.model.MLP_layers.parameters():
            param.requires_grad = False
        for epoch in range(self.local_epochs):
            total_loss=0
            for i, batch in enumerate(self.datas.load_data_i_client(self.id)):
                eta = self.eta
                self.extractor_optimizer.zero_grad()
                labels = torch.tensor(batch.labels).to(self.device)
                d1_model_result = self.model(batch, self.cross_lingual_model, logit=True,e_reps=True)
                d1_predictive_loss = self.loss(F.log_softmax(d1_model_result['logit'],dim=1), labels)
                d2_model_result = self.student_model.forward_to_classify(d1_model_result['feature'], logit=True)
                d2_predictive_loss = self.loss(F.log_softmax(d2_model_result['logit'], dim=1), labels)
                diversity_loss = self.ensemble_loss(F.log_softmax(d2_model_result['logit'],dim=1),F.softmax(d1_model_result['logit'],dim=1))
                loss = d1_predictive_loss + d2_predictive_loss + eta * diversity_loss
                if regularization and epoch < early_stop:
                    z_glob = self.global_extractor(batch, self.cross_lingual_model)
                    z_prev = self.old_extractor(batch,self.cross_lingual_model)
                    z_now = d1_model_result['feature']
                    posi = self.cos(z_now, z_glob)
                    logits = posi.reshape(-1, 1)
                    nega = self.cos(z_now, z_prev)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                    logits /= self.temperature
                    labels = torch.zeros(z_now.size(0)).cuda().long()
                    loss_contrasive = self.mu * self.ce_loss(logits, labels)
                    loss+=loss_contrasive
                total_loss += loss.item()
                loss.backward()
                self.extractor_optimizer.step()  # self.local_model)
            print(f"User {self.id} training extractor on epoch {epoch} loss is : {total_loss / (i + 1):.4f}")
        self.extractor_lr_scheduler.step(glob_iter)
        for param in self.student_model.MLP_layers.parameters():
            param.requires_grad = True
        for param in self.model.MLP_layers.parameters():
            param.requires_grad = True

        self.old_extractor = copy.deepcopy(self.model.extractor)
        for param in self.old_extractor.parameters():
            param.requires_grad = False
        self.train_generator(
            self.gen_batch_size,
            epoches=self.ensemble_epochs // self.n_teacher_iters,
            latent_layer_idx=self.latent_layer_idx,
            verbose=True
        )
        self.model.move_from_device()
        self.student_model.move_from_device()
        self.generative_model.move_from_device(self.model.cpu)
        self.generative_local.move_from_device(self.model.cpu)

    def get_label_weights(self):
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            weights.append(self.label_counts[label])
            if np.max(weights) > 1:
                qualified_labels.append(label)
            # uniform
            label_weights.append( np.array(weights) / np.sum(weights) )
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        return label_weights, qualified_labels

    def train_generator(self, batch_size, epoches=1, latent_layer_idx=-1, verbose=False):
        """
        Learn a generator that find a consensus latent representation z, given a label 'y'.
        :param batch_size:
        :param epoches:
        :param latent_layer_idx: if set to -1 (-2), get latent representation of the last (or 2nd to last) layer.
        :param verbose: print loss information.
        :return: Do not return anything.
        """
        #self.generative_regularizer.train()
        self.label_weights, self.qualified_labels = self.get_label_weights()
        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0
        def update_generator_(n_iters, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS):
            self.generative_local.train()
            self.model.eval()
            self.student_model.eval()
            for i in range(n_iters):
                self.generative_local_optimizer.zero_grad()
                y=np.random.choice(self.unique_labels, batch_size)
                y_input=torch.LongTensor(y).to(self.device)
                ## feed to generator
                gen_result=self.generative_local(y_input, latent_layer_idx=latent_layer_idx, verbose=True)
                # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
                gen_output, eps=gen_result['output'], gen_result['eps']
                ##### get losses ####
                diversity_loss=self.generative_local.diversity_loss(eps, gen_output)  # encourage different outputs

                ######### get teacher loss ############
                self.model.eval()
                user_result_given_gen=self.model.forward_to_classify(gen_output, logit=True)
                teacher_loss= torch.mean(self.generative_local.crossentropy_loss(user_result_given_gen['output'], y_input))
                teacher_logit=user_result_given_gen['logit']

                ######### get student loss ############

                student_output= self.student_model.forward_to_classify(gen_output, logit=True)
                teacher_loss += torch.mean( self.generative_local.crossentropy_loss(student_output['output'], y_input))
                # teacher_loss += teacher_loss_
                generator_div_loss = F.kl_div(F.log_softmax(student_output['logit'],dim=1),F.softmax(teacher_logit,dim=1)) #
                if generator_div_loss.item()!=0 and self.ensemble_beta > 0:
                    loss = self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss - self.ensemble_beta * generator_div_loss #
                else:
                    loss = self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss
                loss.backward()
                self.generative_local_optimizer.step()
                TEACHER_LOSS += self.ensemble_alpha * teacher_loss#
                STUDENT_LOSS += self.ensemble_beta * generator_div_loss#
                DIVERSITY_LOSS += self.ensemble_eta * diversity_loss#
            return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS

        for i in range(epoches):
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS=update_generator_(
                self.n_teacher_iters, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)

        TEACHER_LOSS = TEACHER_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * epoches)
        STUDENT_LOSS = STUDENT_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * epoches)
        DIVERSITY_LOSS = DIVERSITY_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * epoches)
        info="User {} Generator: Teacher Loss= {:.4f}, Student Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
            format(self.id,TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
        if verbose:
            print(info)
        self.generative_lr_scheduler.step()


    def adjust_weights(self, samples):
        labels, counts = samples['labels'], samples['counts']
        np_y = samples['y'].detach().numpy()
        n_labels = samples['y'].shape[0]
        weights = np.array([n_labels / count for count in counts]) # smaller count --> larger weight
        weights = len(self.available_labels) * weights / np.sum(weights) # normalized
        label_weights = np.ones(self.unique_labels)
        label_weights[labels] = weights
        sample_weights = label_weights[np_y]
        return sample_weights

    def reinit_optimizer(self,model,use_adam):
        if use_adam:
            optimizer=torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.learning_rate, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=1e-2, amsgrad=False)
        else:
            optimizer = pFedIBOptimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate)
        return optimizer
