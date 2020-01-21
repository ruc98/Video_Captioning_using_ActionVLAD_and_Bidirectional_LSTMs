import numpy as np
import os
import torch
import torch.utils.data as data_utils
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torchnlp.metrics import get_moses_multi_bleu

torch.cuda.set_device(0)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

torch.manual_seed(1)
np.random.seed(0)

datapath = 'video_data/'
cap_train= np.load(datapath+'captionsTrain.npy',encoding='bytes')
cap_train.shape

def prepare_sequence(seq, to_ix):  # gives index to each word and forms a list of indices
  idxs = [to_ix[w] for w in seq]
  return torch.tensor(idxs, dtype=torch.long).cuda()

word_to_ix = {}    # vocabulary with index values
caplens = np.zeros((cap_train.shape[0],cap_train.shape[1]))

### Creating Vocabulary ###
for i in range(cap_train.shape[0]):
    for j in range(cap_train.shape[1]):
        cap_str = str(cap_train[i,j])
        cap2 = cap_str[2:-1].split()
        caplens[i][j] = len(cap2)

        for word in cap2:
            if word not in word_to_ix:
                word_to_ix[word]=len(word_to_ix)
print(len(word_to_ix))

feat_train = np.load('vgg_feat_train.npy')
train_feat = feat_train[:,:,:500,:,:]
train_feat.shape

class ACTIONVLAD(nn.Module):
    ### ACTIONVLAD layer implementation ###

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):

        super(ACTIONVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha  #float, Parameter of initialization. Larger value is harder assignment.
        self.normalize_input = normalize_input  #Bool,If true, descriptor-wise L2 normalization is applied to input.
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x1):
        vlad_sum=torch.zeros([1,self.dim*self.num_clusters])
        for t in range(30):
            xt = x1[t,:,:]
            x = xt.unsqueeze(0)
            N, C = x.shape[:2]

            if self.normalize_input:
                x = F.normalize(x, p=2, dim=1)  # across descriptor dim
            # weights a_ij
            soft_assign = self.conv(x).view(N, self.num_clusters, -1)
            soft_assign = F.softmax(soft_assign, dim=1)

            x_flatten = x.view(N, C, -1)

            # calculate residuals to each clusters
            residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) -                 self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign.unsqueeze(2)
            vlad = residual.sum(dim=-1)

            vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
            vlad = vlad.view(x.size(0), -1)  # flatten
            vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
            vlad_sum+=vlad
        return vlad_sum

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # define the properties
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
    
        # embedding layer
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, captions, features):
        features = torch.tensor(features[np.newaxis,:],dtype=torch.float32)
        batch_size = 1
        
        # init the hidden and cell states to zeros
        hidden_state = torch.zeros((batch_size, self.hidden_size))
        cell_state = torch.zeros((batch_size, self.hidden_size))
        outputs = torch.empty((batch_size, captions.size(0), self.vocab_size))
        # embed the captions
        captions_embed = self.embed(captions)
        # pass the caption word by word
        for t in range(captions.size(0)):

            # for the first time step the input is the feature vector
            if t == 0:
                hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))
                
            # for the 2nd+ time step, using teacher forcer
            else:
                hidden_state, cell_state = self.lstm_cell(captions_embed[t-1, :].view(1,-1), (hidden_state, cell_state))
            
            out = self.fc_out(hidden_state)
            outputs[:,t, :] = out
    
        return outputs


### Variables ###
features = torch.Tensor(train_feat).cuda()
EMBEDDING_DIM = 4000
HIDDEN_DIM = 64
vocab_size=len(word_to_ix)


### Training ###
model = DecoderRNN(EMBEDDING_DIM, HIDDEN_DIM, vocab_size, num_layers=1)
vladmodel = ACTIONVLAD(num_clusters=8, dim=500, alpha=100.0)
model.cuda()
vladmodel.cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(100):
    loss_epoch=0
    for i in range(features.shape[0]):
        for j in range(15):
            # caption to list of words
            cap_str = str(cap_train[i,j])
            cap2 = cap_str[2:-1].split()
            sentence = cap2
            caplen = caplens[i][j]
            model.zero_grad()
            vladmodel.zero_grad()
            # index of words
            cap_target = prepare_sequence(sentence, word_to_ix)
            avlad = vladmodel(features[i])
            avlad = avlad.view(-1)
            cap_pred = model(cap_target,avlad)
            cap_pred = cap_pred.view(-1,vocab_size)
            loss = loss_function(cap_pred, cap_target.contiguous().view(-1))
            loss.backward()
            loss_epoch+=loss.data.item()
            optimizer.step()
    print(epoch,'  ',loss_epoch)

torch.save(model.state_dict(), './sim_vgg.pth')


### Evaluation of training set ###
model.eval()
vladmodel.eval()
hypothesis=[]
reference=[]
for i in range(cap_train.shape[0]):
    for j in range(cap_train.shape[1]):
        cap_str = str(cap_train[i,j])
        cap2 = cap_str[2:-1].split()
        sentence = cap2
        model.zero_grad()
        vladmodel.zero_grad()
        reference.append(" ".join(sentence))

        cap_target = prepare_sequence(sentence, word_to_ix)
#             print(features.shape)
        avlad = vladmodel(features[i])
#             print(avlad.shape)
        avlad = avlad.view(-1)
        cap_pred = model(cap_target,avlad)
        cap_pred = cap_pred.view(-1,vocab_size)
        maxval, maxidx = torch.max(cap_pred,dim=-1)
        sent=[]
        key_list=list(word_to_ix.keys())    
        for k in range(maxidx.shape[0]):
            sent.append(key_list[maxidx[k]])
        
        hypothesis.append(" ".join(sent))
            
print('bleu-score')
print(get_moses_multi_bleu(hypothesis, reference, lowercase=True))


### Validation set evaluation ###
datapath = 'video_data/'
cap_val= np.load(datapath+'captionsDev.npy',encoding='bytes')
cap_val.shape

feat_val = np.load('vgg_feat_val.npy')
val_feat = feat_val[:,:,:500,:,:]
val_feat.shape
features_val = torch.Tensor(val_feat).cuda()

def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        # if word is not present, assign it random value of 0
        if w not in to_ix:
            idxs.append(0)
        else:
            idxs.append(to_ix[w])
    return torch.tensor(idxs, dtype=torch.long).cuda()

model.eval()
vladmodel.eval()
hypothesis=[]
reference=[]
loss_epoch=0
for i in range(cap_val.shape[0]):
    for j in range(cap_val.shape[1]):
        cap_str = str(cap_val[i,j])
        cap2 = cap_str[2:-1].split()
        sentence = cap2
        model.zero_grad()
        vladmodel.zero_grad()
        reference.append(" ".join(sentence))

        cap_target = prepare_sequence(sentence, word_to_ix)
        avlad = vladmodel(features_val[i])
        avlad = avlad.view(-1)
        cap_pred = model(cap_target,avlad)
        cap_pred = cap_pred.view(-1,vocab_size)
        maxval, maxidx = torch.max(cap_pred,dim=-1)
        sent=[]
        key_list=list(word_to_ix.keys())    
        for k in range(maxidx.shape[0]):
            sent.append(key_list[maxidx[k]])
        
        hypothesis.append(" ".join(sent))
            
print('bleu-score')
print(get_moses_multi_bleu(hypothesis, reference, lowercase=True))


### test set Evaluation ###

datapath = 'video_data/'
cap_test= np.load(datapath+'captionsTest.npy',encoding='bytes')
cap_test.shape

feat_test = np.load('vgg_feat_test.npy')
test_feat = feat_test[:,:,:500,:,:]
test_feat.shape
features_test = torch.Tensor(test_feat).cuda()

model.eval()
vladmodel.eval()
hypothesis=[]
reference=[]
loss_epoch=0
for i in range(cap_test.shape[0]):
    for j in range(cap_test.shape[1]):
        cap_str = str(cap_test[i,j])
        cap2 = cap_str[2:-1].split()
        sentence = cap2
        model.zero_grad()
        vladmodel.zero_grad()
        reference.append(" ".join(sentence))

        cap_target = prepare_sequence(sentence, word_to_ix)
        avlad = vladmodel(features_test[i])
        avlad = avlad.view(-1)
        cap_pred = model(cap_target,avlad)
        cap_pred = cap_pred.view(-1,vocab_size)
        maxval, maxidx = torch.max(cap_pred,dim=-1)
        sent=[]
        key_list=list(word_to_ix.keys())    
        for k in range(maxidx.shape[0]):
            sent.append(key_list[maxidx[k]])
        
        hypothesis.append(" ".join(sent))
            
print('bleu-score')
print(get_moses_multi_bleu(hypothesis, reference, lowercase=True))


### Visualize sentences ###
for i in range(10):
    for j in range(3,4):
        cap_str = str(cap_val[i,j])
        cap2 = cap_str[2:-1].split()
        sentence = cap2
        model.zero_grad()
        vladmodel.zero_grad()
        
        cap_target = prepare_sequence(sentence, word_to_ix)
        avlad = vladmodel(features_test[i])
        avlad = avlad.view(-1)
        cap_pred = model(cap_target,avlad)
        cap_pred = cap_pred.view(-1,vocab_size)
        maxval, maxidx = torch.max(cap_pred,dim=-1)
        sent=[]
        key_list=list(word_to_ix.keys())    
        for k in range(maxidx.shape[0]):
            sent.append(key_list[maxidx[k]])
        sent_targ = []
        sent_pred = []
        sent_targ.append(" ".join(sentence))
        sent_pred.append(" ".join(sent))

        print(sent_targ)
        print(sent_pred)
        print(" ")
print('DONE!!!')
