import torch
from FLAlgorithms.users.userbase import User

class UserAVG(User):
    def __init__(self,  args, id, model, dataset,device, use_adam=True,cross_lingual_model=None):
        super().__init__(args, id, model, dataset, use_adam=use_adam)
        self.device= device
        self.cross_lingual_model= cross_lingual_model

    def update_label_counts(self, labels):
        for label in labels:
            self.label_counts[int(label)] += 1

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {int(label):1 for label in range(self.unique_labels)}

    def test_personalized_model(self):
        self.model.to_device()
        self.model.eval()
        test_acc = 0
        loss = 0
        # self.update_parameters(self.personalized_model_bar)
        for i, batch in enumerate(self.datas.load_data_i_client("test")):
            with torch.no_grad():
                output = self.model(batch, self.cross_lingual_model)['output']
                labels = torch.tensor(batch.labels).to(self.device)
                loss += self.ce_loss(output, labels).item()
                test_acc += (torch.sum(torch.argmax(output, dim=1) == labels)).item()
        # @loss += self.loss(output, y)
            #@loss += self.loss(output, y)
            #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            #print(self.id + ", Test Loss:", loss)
        self.model.move_from_device()
        return test_acc,self.datas.test_numbers, loss

    def train(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        self.clean_up_counts()
        self.model.to_device()
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            total_loss = 0
            for i, batch in enumerate(self.datas.load_data_i_client(self.id)):
                labels = batch.labels
                model_result = self.model(batch, self.cross_lingual_model, logit=True)
                labels = torch.tensor(batch.labels).to(self.device)
                user_output_logp = model_result['output']
                if count_labels:
                    self.update_label_counts(labels)
                self.optimizer.zero_grad()
                loss=self.ce_loss(user_output_logp, labels)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()#self.plot_Celeb)
            print(f"User {self.id} training model on epoch {epoch} loss is : {total_loss / (i + 1):.4f}")
            # local-model <=== self.model
            # self.clone_model_paramenter(self.model.parameters(), self.local_model)
            # if personalized:
            #     self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
            # local-model ===> self.model
            #self.clone_model_paramenter(self.local_model, self.model.parameters())
        if lr_decay:
            self.lr_scheduler.step(glob_iter)

        self.model.move_from_device()
