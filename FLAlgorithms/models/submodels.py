import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.transformer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads, hidden_size, dropout, ff_size):
        self.num_layers = num_layers

        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(hidden_size, dropout, num_heads, ff_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, src, src_pad):
        src_mask = src_pad.unsqueeze(1)
        for i in range(self.num_layers):
            output = self.layers[i](src, src_mask)
        return self.norm(output)

# Attention Mechanism in a nutshell
# Input : impending vectors of size vector_num * vector_size
# Output : Attentioned representation of size vector_size
class Attention(nn.Module):

    def __init__(self, vector_size):
        super(Attention, self).__init__()
        self.vector_size = vector_size
        self.fc = nn.Linear(vector_size, vector_size)
        self.weightparam = nn.Parameter(torch.randn(vector_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, vectors):
        weight = torch.tanh(self.fc(vectors)).matmul(self.weightparam)
        weight = F.softmax(weight, dim=0)
        rep = vectors.mul(weight)
        del weight
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        rep = rep.sum(dim=0)
        return rep

    def forward_variant(self, vectors,indexes):
        weight = torch.tanh(self.fc(vectors)).matmul(self.weightparam)
        reps = []
        # weights = []
        for i,x in enumerate(indexes[:-1]):
            # TweetLevelRep = self.TweetHighAttention(TweetRep[x:indexes[i+1]])  # 这个地方可以进行优化加速
            # tweet_reps.append(TweetLevelRep)
            weight_tmp = F.softmax(weight[x:indexes[i + 1]], dim=0)
            rep = vectors[x:indexes[i + 1]].mul(weight_tmp)
            rep = rep.sum(dim=0)
            reps.append(rep)
        return reps


# Model for word-level semantics extraction (2 initializations: 1 WordLevel(batch_size of 1) 1 TweetLevelLow)
# Input : sequence of size seq_len * batch_size (maybe a result of rnn.pad_sequence?)
# Output : seq_len * batch_size * rep_size
class SemanticWord(nn.Module):

    def __init__(self, embedding_dim,hidden_size,bidirectional, batch_size, num_layer, p,encoder_type,config=None):
        super(SemanticWord, self).__init__()
        self.hidden_dim = hidden_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.num_layer = num_layer
        self.encoder_type = encoder_type
        if self.encoder_type == "LSTM":
            self.encoder = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=bidirectional,batch_first=True,dropout=p, num_layers=num_layer)
            self.forward = self.forward_LSTM
        else:
            self.encoder = Encoder(config['layers'],
                          config['heads'],
                          config['hidden_size'],
                          config['dropout'],
                          config['ff_size'],)
            self.forward = self.forward_transformer

    def init_hidden(self, batch_size):
        # torch.cuda.FloatTensor(1000, 1000).fill_(0)
        temp = (torch.cuda.FloatTensor(2 * self.num_layer, batch_size, self.hidden_dim).fill_(0),
                torch.cuda.FloatTensor(2 * self.num_layer, batch_size, self.hidden_dim).fill_(0))
        return temp
        # return (temp[0].type(torch.FloatTensor).cuda(non_blocking=True), temp[1].type(torch.FloatTensor).cuda(non_blocking=True))

    def forward_transformer(self,text,padded):
        sim_batch_size = 200
        batch_size = len(text)
        if batch_size<sim_batch_size:
            out = self.encoder(text,padded)
            return out
        else:
            now = 0
            tmp = []
            # print('batch_size: ' + str(batch_size))
            while True:
                now_text = text[now:min(now + sim_batch_size, batch_size), :]
                now_batch_size = len(now_text)
                now_padded = padded[now:now+now_batch_size]
                out = self.encoder(now_text, now_padded)
                # del self.hidden
                del now_text,now_padded
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                tmp.append(out)
                if now + sim_batch_size >= batch_size:
                    break
                now += sim_batch_size
            tmp = torch.cat(tmp, dim=0)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # print('before attention: ', tmp.size())
            return tmp

    def forward_LSTM(self, text,padded=None):   # for lstm
        sim_batch_size = 200
        batch_size = len(text)
        if batch_size <= sim_batch_size:
            self.hidden = self.init_hidden(batch_size)
            # text = text[0:text.detach().tolist().index(tdict['b_b'])]

            result, _ = self.encoder(text, self.hidden)
            del self.hidden
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return result
        else:
            now = 0
            tmp = []
            # print('batch_size: ' + str(batch_size))
            while True:
                now_text = text[now:min(now + sim_batch_size, batch_size),: ]
                now_batch_size = len(now_text)
                # print('now batch size: ' + str(now_batch_size))
                self.hidden = self.init_hidden(now_batch_size)
                # result = result.clone().view(len(now_text), now_batch_size, -1).cuda(non_blocking=True)
                result, _ = self.encoder(now_text, self.hidden)
                # del self.hidden
                del now_text
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                tmp.append(result)
                if now + sim_batch_size >= batch_size:
                    break
                now += sim_batch_size
            tmp = torch.cat(tmp, dim=0)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # print('before attention: ', tmp.size())
            return tmp

# Model for tweet-level semantics extraction from tweet vectors
# Input : sequence of tweet vectors of a single user of size vector_num * 1 * tweet_vec_size
# Output : vector_num * rep_size
class SemanticTweet(nn.Module):

    def __init__(self, input_dim,hidden_size,bidirectional, rep_size, num_layer, p,encoder_type,encoder=None,device=None):
        super(SemanticTweet, self).__init__()
        self.hidden_dim = hidden_size
        self.input_dim = input_dim
        self.rep_size = rep_size
        self.batch_size = 1
        self.num_layer = num_layer
        self.encoder_type = encoder_type
        self.device =device

        if self.encoder_type == "LSTM":
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, bidirectional=bidirectional, dropout=p,
                                num_layers=num_layer)
            self.hidden = self.init_hidden()
            self.forward = self.forward_LSTM
        else:
            self.encoder = encoder
            self.forward = self.forward_transformer

    def init_hidden(self):
        temp = (torch.cuda.FloatTensor(2 * self.num_layer, self.batch_size, self.hidden_dim).fill_(0),
                torch.cuda.FloatTensor(2 * self.num_layer, self.batch_size, self.hidden_dim).fill_(0))
        return temp
    def forward_transformer(self,vectors):
        vectors= vectors.transpose(0, 1)
        pad = torch.zeros((vectors.size()[0],vectors.size()[1])).bool().to(self.device)
        result = self.encoder(vectors,pad)
        del vectors, pad
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        result = result.squeeze(0)
        return result

    def forward_LSTM(self, vectors):
        self.hidden = self.init_hidden()
        result, _ = self.encoder(vectors, self.hidden)
        result = result.squeeze(1)
        del self.hidden
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return result


class TweetCNNExtractor(nn.ModuleList):

    def __init__(self,embedding_dim,hidden_size, p,encoder_type,output_size,config=None):
        super(TweetCNNExtractor, self).__init__()

        # Parameters regarding text preprocessing
        # self.seq_len = params.seq_len
        self.embedding_size = embedding_dim

        # Dropout definition
        self.dropout = nn.Dropout(p)

        # CNN parameters definition
        # Kernel sizes
        self.kernel_1 = config["kernel_size"][0]
        self.kernel_2 = config["kernel_size"][1]
        self.kernel_3 = config["kernel_size"][2]
        self.kernel_4 = config["kernel_size"][3]

        # Output size for each convolution
        self.out_size = config["outsize"]
        # Number of strides for each convolution
        self.stride = config["stride"]
        self.seq_len = config["seq_len"]
        # Embedding layer definition
        # Convolution layers definition
        self.conv_1 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_1, self.stride)
        self.conv_2 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_2, self.stride)
        self.conv_3 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_3, self.stride)
        self.conv_4 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_4, self.stride)

        # Max pooling layers definition
        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)

        # Fully connected layer definition
        self.fc = nn.Linear(self.in_features_fc(), self.embedding_size)
        self.reset_parameters()

    def in_features_fc(self):
        '''Calculates the number of output features after Convolution + Max pooling

        Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
        Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1

        source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        '''
        # Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
        out_conv_1 = ((self.seq_len - self.kernel_1 ) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = (out_conv_1 - self.kernel_1)/ self.stride+ 1
        out_pool_1 = math.floor(out_pool_1)

        # Calcualte size of convolved/pooled features for convolution_2/max_pooling_2 features
        out_conv_2 = ((self.seq_len - self.kernel_2) / self.stride) + 1
        out_conv_2 = math.floor(out_conv_2)
        out_pool_2 = (out_conv_2 - self.kernel_2)/self.stride+ 1
        out_pool_2 = math.floor(out_pool_2)

        # Calcualte size of convolved/pooled features for convolution_3/max_pooling_3 features
        out_conv_3 = ((self.seq_len - self.kernel_3) / self.stride) + 1
        out_conv_3 = math.floor(out_conv_3)
        out_pool_3 = (out_conv_3 - self.kernel_3)/self.stride+ 1
        out_pool_3 = math.floor(out_pool_3)

        # Calcualte size of convolved/pooled features for convolution_4/max_pooling_4 features
        out_conv_4 = ((self.seq_len - self.kernel_4) / self.stride) + 1
        out_conv_4 = math.floor(out_conv_4)
        out_pool_4 = (out_conv_4 - self.kernel_4)/self.stride+ 1
        out_pool_4 = math.floor(out_pool_4)

        # Returns "flattened" vector (input for fully connected layer)
        return (out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4) * self.out_size
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        # Sequence of tokes is filterd through an embedding layer
        x = x.permute(0, 2, 1)
        # Convolution layer 1 is applied
        x1 = self.conv_1(x)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)

        # Convolution layer 2 is applied
        x2 = self.conv_2(x)
        x2 = torch.relu((x2))
        x2 = self.pool_2(x2)

        # Convolution layer 3 is applied
        x3 = self.conv_3(x)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)

        # Convolution layer 4 is applied
        x4 = self.conv_4(x)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)

        # The output of each convolutional layer is concatenated into a unique vector
        union = torch.cat((x1, x2, x3, x4), 2)
        union = union.reshape(union.size(0), -1)

        # The "flattened" vector is passed through a fully connected layer
        out = self.fc(union)
        # Dropout is applied
        out = self.dropout(out)
        # Activation function is applied
        out = torch.relu(out)

        return out.squeeze()

# Model for transforming the properties vector
# Input : property vectors of size vector_num * input_size
# Output : representations of size vector_num * rep_size
class Properties(nn.Module):

    def __init__(self, input_size, rep_size, dropout):
        super(Properties, self).__init__()
        self.input_size = input_size
        self.rep_size = rep_size

        self.fc1 = nn.Linear(self.input_size, self.input_size)
        # self.fc2 = nn.Linear(self.input_size, self.rep_size)
        # self.fc3 = nn.Linear(self.rep_size, self.rep_size)
        self.fc1.bias.data.fill_(0)
        # self.fc2.bias.data.fill_(0)
        # self.fc3.bias.data.fill_(0)
        self.act1 = nn.ReLU()
        # self.act2 = nn.ReLU()
        # self.act3 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        # self.dropout2 = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.input_size, self.rep_size)
        self.act = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, vectors):
        vectors = self.dropout1(self.act1(self.fc1(vectors)))
        # vectors = self.dropout2(self.act2(self.fc2(vectors)))
        # vectors = self.act3(self.fc3(vectors))
        vectors = self.act(self.fc(vectors))
        return vectors

