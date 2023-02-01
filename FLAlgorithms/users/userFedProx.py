import torch
from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import FedProxOptimizer

class UserFedProx(User):
    def __init__(self, args, id, model, dataset,device, use_adam=False,cross_lingual_model=None):
        super().__init__(args, id, model, dataset, use_adam=use_adam)
        self.device = device
        self.cross_lingual_model = cross_lingual_model
        self.model.to_device()
        self.optimizer = FedProxOptimizer(self.model.parameters(), lr=self.learning_rate, lamda=self.lamda)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)

    def update_label_counts(self, labels):
        for label in labels:
            self.label_counts[int(label)] += 1

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {int(label):1 for label in range(self.unique_labels)}

    def train(self, glob_iter, lr_decay=True, count_labels=False):
        self.clean_up_counts()
        self.model.train()
        # cache global model initialized value to local model
        self.clone_model_paramenter(self.local_model, self.model.parameters())
        self.local_model = [ param.to(self.device) for param in self.local_model]
        self.model.to_device()

        for epoch in range(self.local_epochs):
            self.model.train()
            total_loss = 0
            # for i in range(self.K):
            for i, batch in enumerate(self.datas.load_data_i_client(self.id)):
                labels = batch.labels
                if count_labels:
                    self.update_label_counts(labels)

                self.optimizer.zero_grad()
                model_result=self.model(batch,self.cross_lingual_model, logit=True)
                labels = torch.tensor(batch.labels).to(self.device)
                user_output_logp = model_result['output']
                loss=self.ce_loss(user_output_logp, labels)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step(self.local_model)
            print(f"User {self.id} training model on epoch {epoch} loss is : {total_loss / (i + 1):.4f}")
        if lr_decay:
            self.lr_scheduler.step(glob_iter)
        self.model.move_from_device()
        # for param in self.local_model:
        #     param.to(self.model.cpu)