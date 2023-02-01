import torch.nn as nn
import torch.nn.functional as F
from FLAlgorithms.utils.model_config import CONFIGS_
from .submodels import Attention,SemanticWord,SemanticTweet,Properties,TweetCNNExtractor
import torch
#################################
##### Neural Network model #####
#################################
class ExtractNet(nn.Module):
    def __init__(self,config,device,model_type,only_properties=True,visualize_boundary=False):
        super(ExtractNet, self).__init__()
        self.device = device
        self.input_size, self.hidden_size, self.output_size, self.num_layers, self.dropout_prob, self.bidirectional, self.num_mlp_layers,_,self.property_size  = config
        self.encoder_type = model_type
        self.visualize_boundary = visualize_boundary
        if self.encoder_type =="transformer":
            self.transformer_config = CONFIGS_["transformer_config"]
        self.only_properties = only_properties
        self.build_network()

    def build_network(self):
        if self.encoder_type == "transformer":
            config = {'layers':self.transformer_config[0],'heads':self.transformer_config[1],'hidden_size':self.transformer_config[2],'ff_size':self.transformer_config[3],'dropout':self.transformer_config[4]}
            self.encode_hidden = config['hidden_size']
        elif self.encoder_type == "cnn":
            config = CONFIGS_["cnn_config"]
            # num_convs,kernel_size,stride,outsize,drop_out,sqe_len
            config = {'num_convs':config[0],'kernel_size':config[1],'stride':config[2],'hidden_size':config[3],'outsize':config[4],'drop_out':config[5],'seq_len':config[6]}
            self.encode_hidden = config['hidden_size']
        else:
            num=1
            if self.bidirectional:
                num=2
            config = None
            self.encode_hidden = self.hidden_size *num

        self.PropertyModel = Properties(input_size=self.property_size,dropout=self.dropout_prob,rep_size=self.output_size)
        if self.only_properties:
            self.fusion_layer = nn.Linear(self.output_size,self.output_size)
            if self.visualize_boundary:
                self.fusion_layer = nn.Linear(self.output_size,2)
            self.forward = self.foward_only_properties
        else:
            if self.encoder_type == "transformer":
                self.TweetLowModel = SemanticWord(embedding_dim=self.input_size, hidden_size=self.hidden_size,
                                              bidirectional=self.bidirectional,
                                              batch_size=1,
                                              num_layer=self.num_layers, p=self.dropout_prob,
                                              encoder_type=self.encoder_type, config=config)
                self.TweetHighModel = SemanticTweet(input_dim=self.input_size, hidden_size=self.hidden_size,
                                                bidirectional=self.bidirectional, rep_size=self.output_size,
                                                num_layer=self.num_layers, p=self.dropout_prob,
                                                encoder_type=self.encoder_type, encoder=self.TweetLowModel.encoder,
                                                device=self.device)
                self.TweetLowAttention = Attention(vector_size=self.encode_hidden)
                self.TweetHighAttention = Attention(vector_size=self.encode_hidden)
                self.fusion_layer = nn.Linear(self.input_size + self.output_size, self.output_size)
                self.forward = self.foward_transformer
            elif self.encoder_type == "cnn":
                self.Tweetextractor = TweetCNNExtractor(embedding_dim=self.input_size, hidden_size=self.hidden_size, p=self.dropout_prob,
                                              encoder_type=self.encoder_type,output_size = self.output_size, config=config)
                self.TweetHighAttention = Attention(vector_size=self.encode_hidden)
                self.fusion_layer = nn.Linear(self.input_size + self.output_size, self.output_size)
                if self.visualize_boundary:
                    self.fusion_layer = nn.Linear(self.input_size + self.output_size, 2)
                self.forward = self.foward_CNN_attention
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fusion_layer.weight)

    def foward_only_properties(self,batch_users,lingual_model):
        properties = batch_users.properties
        property_reps = self.PropertyModel(torch.FloatTensor(properties).to(self.device))
        user_reps = property_reps
        out = self.fusion_layer(user_reps)
        return out

    def foward_CNN_attention(self,batch_users,lingual_model):
        properties = batch_users.properties
        property_reps = self.PropertyModel(torch.FloatTensor(properties).to(self.device))
        tweets = batch_users.tweets
        tweet_reps = []
        all_tweets=[]
        indexes =[0]
        for i,user_tweet in enumerate(tweets):
            user_tweet = user_tweet[:20]
            all_tweets.append(user_tweet)
            indexes.append(user_tweet.shape[0] + indexes[i])
        all_tweets = torch.cat(all_tweets,dim=0)
        all_tweets, padded = lingual_model.forward_to_embedding(all_tweets.to(self.device))
        TweetRep = self.Tweetextractor(all_tweets)
        tweet_reps = self.TweetHighAttention.forward_variant(TweetRep,indexes)
        tweet_reps = torch.stack(tweet_reps)
        # # property vector extraction
        user_reps = torch.cat((property_reps, tweet_reps), dim=-1)
        out = self.fusion_layer(user_reps)
        return out


    def foward_transformer(self,batch_users,lingual_model):
        """
                Perform a forward pass of our model for feature extraction.
                """
        properties = batch_users.properties
        property_reps = self.PropertyModel(torch.FloatTensor(properties).to(self.device))
        tweets = batch_users.tweets
        tweet_reps = []
        for user_tweet in tweets:
            user_tweet, padded = lingual_model.forward_to_embedding(user_tweet.to(self.device))
            TweetRep = self.TweetLowAttention(
                    self.TweetLowModel(user_tweet, padded.to(self.device)).transpose(0, 1)).unsqueeze(1)
            TweetLevelRep = self.TweetHighAttention(self.TweetHighModel(TweetRep))  # 这个地方可以进行优化加速
            tweet_reps.append(TweetLevelRep)
            del user_tweet, TweetRep, TweetLevelRep
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        tweet_reps = torch.stack(tweet_reps)
            # # property vector extraction

        user_reps = torch.cat((property_reps, tweet_reps), dim=-1)
        out = self.fusion_layer(user_reps)
        return out

class TeacherNet(nn.Module):
    """
    The teacher net
    """

    def __init__(self,model, dataset,device=None,only_properties=True,visualize_boundary=False):
        """
        Initialize the model by setting up the layers.
        """
        super(TeacherNet, self).__init__()
        # define network layers
        print("Creating model for {}".format(dataset))
        self.cpu = torch.device('cpu')
        self.dataset = dataset
        self.device = device
        self.visualize_boundary = visualize_boundary
        self.input_size,self.hidden_size,self.output_size,self.num_layers,self.dropout_prob,self.bidirectional,self.num_mlp_layers,self.encoder_type,_=CONFIGS_[dataset]
        config = CONFIGS_[dataset]
        self.extractor_model_type=model
        self.only_properties= only_properties
        self.build_network(config)

    def to_device(self):
        self.extractor.to(self.device)
        self.to(self.device)

    def move_from_device(self):
        self.extractor.to(self.cpu)
        self.to(self.cpu)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def build_network(self,config):
        self.extractor = ExtractNet(config,self.device,self.extractor_model_type,only_properties=self.only_properties,visualize_boundary=self.visualize_boundary)
        self.MLP_layers = nn.ModuleList()
        for i in range(self.num_mlp_layers):
            if i==0:
                if self.visualize_boundary:
                    self.MLP_layers += [nn.Linear(2, self.hidden_size)]
                else:
                    self.MLP_layers+=[ nn.Linear(self.output_size, self.hidden_size)]
                continue
            # self.MLP_layers += [nn.BatchNorm1d(self.hidden_size, 0.8)] #最后一个batch只有一个没办法用这个换成layernorm再算一遍
            nn.LeakyReLU(0.2, inplace=True),
            if i < self.num_mlp_layers-1:
                self.MLP_layers += [ nn.Linear(self.hidden_size, self.hidden_size)]
            else:
                self.MLP_layers += [nn.Linear(self.hidden_size,2)]
        # Sigmoid output layer
        self.softmax = F.softmax
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.MLP_layers:
            nn.init.xavier_uniform_(layer.weight)

    def get_shared_parameters(self):
        return self.extractor.parameters()

    def forward(self, batch_users,lingual_model,logit=False,e_reps=False):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        results = {}
        out = self.extractor(batch_users,lingual_model)
        if e_reps:
            results['feature'] = out
        for layer in self.MLP_layers:
            out = layer(out)
        if logit:
            results['logit'] = out
        # Sigmoid function
        out = self.softmax(out,dim=1)
        results['output'] = out
        return results

    def forward_to_classify(self,input,logit=False):
        results = {}
        out = input
        for layer in self.MLP_layers:
            out = layer(out)
        if logit:
            results['logit'] = out
        # Sigmoid function
        out = self.softmax(out,dim=1)
        results['output'] = out
        return results

if __name__=="__main__":
    from data.bot_dataset import SocialbotDataset
    from utils import get_device,parseopt
    from data import build_dataset
    import torch
    import os
    from model import EModel
    from utils import parseopt
    opt = parseopt.parse_train_args()
    checkpoint_num = []
    base_path = os.path.dirname(os.path.dirname(__file__))
    base_path = os.path.dirname(base_path)
    path =base_path+'/'+opt.model_path + "_gpu" + str(opt.gpu) + "_warmup" + str(opt.warm_up) + "_latent" + str(
        opt.latent_dim) + "_kl" + str(opt.kl_annealing_steps) + "_split" + str(opt.split)
    if os.path.exists(path):
        files = os.listdir(path)
        for fil in files:
            if not os.path.isdir(fil) and len(fil) > 20:
                checkpoint_num.append(int(fil.split("-")[-1]))
        if len(checkpoint_num) > 0:
            opt.train_from =base_path+'/'+ opt.model_path + "_gpu" + str(opt.gpu) + "_warmup" + str(opt.warm_up) + "_latent" + str(
                opt.latent_dim) + "_kl" + str(opt.kl_annealing_steps) + "_split" + str(
                opt.split) + "/checkpoint-step-%06d" % max(checkpoint_num)
    device = get_device(1)
    cpu = get_device(0,cpu=True)
    dataset_name = 'botometer-feedback-2019'
    print("Build dataset...")
    train_dataset = build_dataset(opt, opt.train, opt.vocab, device, train=True, testdata=False, V=opt.max_test_data)
    dataset = SocialbotDataset(dataset_name,batch_size=3,device=device,train=True)
    fields = train_dataset.fields
    print("Build model...")
    pad_ids = {"src": fields["src"].pad_id, "tgt": fields["tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab), "tgt": len(fields["tgt"].vocab)}
    emodel = EModel.load_model(opt, device, pad_ids, vocab_sizes).to(device)
    emodel.eval()
    model = TeacherNet(dataset_name,device).to(device)
    epochs = 100
    optimizer = torch.optim.Adam(params=model.parameters(),lr= 1e-4, betas=(0.9, 0.999),eps=1e-08, weight_decay=0, amsgrad=False)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)
    lossf = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        nums = 0
        accs = 0
        for i, batch in enumerate(dataset):
            results = model(batch,emodel,cpu)
            labels = torch.tensor(batch.labels).to(device)#torch.nn.functional.one_hot(torch.tensor(batch.labels),num_classes=2).to(device)
            predict = results['output']
            optimizer.zero_grad()
            loss = lossf(predict,labels.long())
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
            pred = torch.argmax(predict, dim=1).to(cpu)
            acc = sum(pred==torch.tensor(batch.labels)).item()
            nums+=len(pred)
            accs +=acc
            if i % 30 == 0:
                print(f'batch {i}')
        print(f"epoch:{epoch} loss:{total_loss/len(dataset.batches)} acc:{accs/nums}")