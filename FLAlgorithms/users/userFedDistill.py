import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import pFedIBOptimizer

class LogitTracker():
    def __init__(self, unique_labels):
        self.unique_labels = unique_labels
        self.labels = [i for i in range(unique_labels)]
        self.label_counts = torch.ones(unique_labels) # avoid division by zero error
        self.logit_sums = torch.zeros((unique_labels,unique_labels) )

    def update(self, logits, Y):
        """
        update logit tracker.
        :param logits: shape = n_sampls * logit-dimension
        :param Y: shape = n_samples
        :return: nothing
        """
        batch_unique_labels, batch_labels_counts = Y.unique(dim=0, return_counts=True)
        self.label_counts[batch_unique_labels] += batch_labels_counts
        # expand label dimension to be n_samples X logit_dimension
        labels = Y.view(Y.size(0), 1).expand(-1, logits.size(1))
        logit_sums_ = torch.zeros((self.unique_labels, self.unique_labels) )
        logit_sums_.scatter_add_(0, labels, logits)
        self.logit_sums += logit_sums_


    def avg(self):
        res= self.logit_sums / self.label_counts.float().unsqueeze(1)
        return res


class UserFedDistill(User):
    """
    Track and average logit vectors for each label, and share it with server/other users.
    """
    def __init__(self, args, id, model,  dataset,device,  unique_labels, use_adam=False,cross_lingual_model=None):
        super().__init__(args, id, model, dataset, use_adam=use_adam)
        self.device = device
        self.cross_lingual_model = cross_lingual_model
        self.init_loss_fn()
        self.unique_labels = unique_labels
        self.label_counts = {}
        self.logit_tracker = LogitTracker(self.unique_labels)
        self.global_logits = None
        self.reg_alpha = 1

    def update_label_counts(self, labels):
        for label in labels:
            self.label_counts[int(label)] += 1

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {int(label):1 for label in range(self.unique_labels)}

    def train(self, glob_iter, personalized=True, lr_decay=True, count_labels=True, verbose=True):
        self.clean_up_counts()
        self.model.train()
        REG_LOSS, TRAIN_LOSS = 0, 0
        self.model.to_device()
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            total_loss = 0
            for i, batch in enumerate(self.datas.load_data_i_client(self.id)):
                labels = batch.labels
                if count_labels:
                    self.update_label_counts(labels)
                self.optimizer.zero_grad()
                model_result = self.model(batch, self.cross_lingual_model, logit=True)
                labels = torch.tensor(batch.labels).to(self.device)
                output, logit = model_result['output'], model_result['logit']
                self.logit_tracker.update(logit.to(self.model.cpu),torch.tensor(batch.labels))
                if self.global_logits != None:
                    ### get desired logit for each sample
                    train_loss = self.ce_loss(output, labels)
                    self.global_logits.to(self.device)
                    target_p = F.log_softmax(self.global_logits[labels,:], dim=1).to(self.device)
                    reg_loss = self.ensemble_loss(F.log_softmax(model_result['logit'],dim=1).to(self.device), target_p)
                    REG_LOSS += reg_loss
                    TRAIN_LOSS += train_loss
                    loss = train_loss + self.reg_alpha * reg_loss
                else:
                    loss=self.ce_loss(output, labels)
                loss.backward()
                total_loss += loss.item()
                self.optimizer.step()#self.local_model)
            print(f"User {self.id} training model on epoch {epoch} loss is : {total_loss / (i + 1):.4f}")
            # local-model <=== self.model
            self.clone_model_paramenter(self.model.parameters(), self.local_model)
            if personalized:
                self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
        if lr_decay:
            self.lr_scheduler.step(glob_iter)
        if self.global_logits != None and verbose:
            REG_LOSS = REG_LOSS.detach().cpu().numpy() / (self.local_epochs * self.K)
            TRAIN_LOSS = TRAIN_LOSS.detach().cpu().numpy() / (self.local_epochs * self.K)
            info = "Train loss {:.2f}, Regularization loss {:.2f}".format(REG_LOSS, TRAIN_LOSS)
            print(info)



