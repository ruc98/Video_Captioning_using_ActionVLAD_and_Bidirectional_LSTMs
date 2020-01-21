# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fkvgdn9fENqJr6OaYOQEZ9BaVJtdU7Eq
"""

from google.colab import drive
drive.mount('/content/gdrive')

!python3 download_nus_wide.py --url_dir /content --save_dir /content/images --num_threads 1000

!unzip Assignment_2.zip /content/Assignment_2

!cp /content/gdrive/'My Drive'/images.zip images.zip
!unzip images.zip

!zip -r images.zip images

!cp /content/npimdata1.npy /content/gdrive/'My Drive'

!cp /content/gdrive/'My Drive'/npimdata1.npy npimdata1.npy

import csv 
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import matplotlib.image as img
import PIL.Image

fname="Y.csv"
expec=None
with open(fname, 'r') as csvfile: 
  # creating a csv reader object 
  csvreader = csv.reader(csvfile) 
  
  # extracting each data row one by one 
  i=0
  j=0
  for row in csvreader: 
    try:
      i=i+1
      #print(row)
      #print(image)
      #print(image.shape)
    except:
      j=j+1
      pass
  print(i)
  print(j)
  expec=np.zeros(shape=(i,6),dtype=int)
  
with open(fname, 'r') as csvfile: 
  # creating a csv reader object 
  csvreader = csv.reader(csvfile) 
  
  # extracting each data row one by one 
  i=0
  j=0
  for row in csvreader: 
    try:
      for ij in range(6):
        expec[i][ij]=int(row[ij])
        #print(row[ij])
      #print(image)
      #print(image.shape)
      i=i+1
    except:
      j=j+1
      pass
  print(i)
  print(j)

import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from torch.autograd import Variable

batch_size=50
print(feat.shape)
print(targets.shape)
print(targets)
dataset = data_utils.TensorDataset(feat, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

class mlffn(nn.Module):
    def __init__(self):
        super(mlffn, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(red_dim, 10),
            nn.ReLU(True),
            nn.Linear(10, 15),
            nn.ReLU(True),
            nn.Linear(15, 6),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        return x
      
num_epochs=200
classif=mlffn()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    classif.parameters(), lr=0.0001, weight_decay=1e-5)
prev_loss=10000

for epoch in range(num_epochs):
    for data in dataloader:
        #print (data)
        img, clas = data
        img = img.view(img.size(0), -1)
        img = Variable(img)
        #print (img.shape),128,784
        #print (clas)
        #clas=clas.type(torch.IntTensor)
        # ===================forward=====================
        #img = model.bottle(img)
        #print (img.shape)
        output = classif(img)
        #print (output[0][1].item())
        # 128,784
        loss = criterion(output, clas)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # ===================log========================
    loss=criterion(classif(feat),targets)
    corr=0
      
    print('epoch [{}/{}], loss:{:.4f} ,accuracy:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data,corr))
    if prev_loss-loss.data < 0.0001:
    	break
    prev_loss=loss.data
corr=0
tot=0
tr=np.zeros(shape=(6))
to=np.zeros(shape=(6))
feat1=Variable(feat_t)
output=classif(feat1)
targ1=targ_t.type(torch.IntTensor)
for i in range(targ1.shape[0]):
  for j in range(targ1.shape[1]):
    if(output[i][j].item() > 0.5 and targ1[i][j] == 1):
      corr=corr+1
      tr[j]=tr[j]+1
    elif(output[i][j].item() <= 0.5 and targ1[i][j] == 0):
      corr=corr+1
      tr[j]=tr[j]+1
    tot=tot+1

for i in range(6):
  tr[i]=tr[i]/targ1.shape[0]
    
print(tr)
print(output)
print(targ1)
print('accuracy')
print(corr/tot)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

red_dim=15
pca = PCA(n_components=red_dim)
features=pca.fit_transform(imdata1)

#print(pca.explained_variance_ratio_)
su=np.zeros(shape=(imdata1.shape[1]))
nu=np.zeros(shape=(imdata1.shape[1]))
su1=0
j=0
for i in pca.explained_variance_ratio_:
  nu[j]=j+1
  su1=su1+i
  su[j]=su1
  j=j+1
  

plt.plot(nu,su)
plt.plot(red_dim,su[red_dim],'-o')
#plt.annotate('({},{:.4f})'.format(red_dim,err[red_dim]),(red_dim,err[red_dim]))
#print(err)
#plt.plot(170,err[170],'-o')
#plt.annotate('({},{:.4f})'.format(170,err[170]),(170,err[170]))
plt.xlabel('Number of Principle Components')
plt.ylabel('Variance Explained')
#plt.show()

plt.plot(nu[:4096],su[:4096])
plt.plot(red_dim,su[red_dim],'-o')
#plt.annotate('({},{:.4f})'.format(red_dim,err[red_dim]),(red_dim,err[red_dim]))
#print(err)
#plt.plot(170,err[170],'-o')
#plt.annotate('({},{:.4f})'.format(170,err[170]),(170,err[170]))
plt.xlabel('Number of Principle Components')
plt.ylabel('Variance Explained')

print(feat.shape)
print(expec.shape)

print(features)

im_test=features[:int(imdata.shape[0]*0.3),:]
tar_test=expec[:int(imdata.shape[0]*0.3),:]
features=features[int(imdata.shape[0]*0.3):,:]
expec=expec[int(imdata.shape[0]*0.3):,:]
feat_t=torch.tensor(im_test,dtype=torch.float32)
targ_t=torch.tensor(tar_test,dtype=torch.float32)
targets=torch.tensor(expec,dtype=torch.float32)
feat=torch.tensor(features,dtype=torch.float32)

im_test=features[:int(imdata1.shape[0]*0.3),:]
tar_test=expec[:int(imdata1.shape[0]*0.3),:]
features=features[int(imdata1.shape[0]*0.3):,:]
expec=expec[int(imdata1.shape[0]*0.3):,:]
feat_t=torch.tensor(im_test,dtype=torch.float32)
targ_t=torch.tensor(tar_test,dtype=torch.float32)
targets=torch.tensor(expec,dtype=torch.float32)
feat=torch.tensor(features,dtype=torch.float32)

print(imdata1)

np.save('npimdata1',imdata1)

np.save('npimdata',imdata)

!cp /content/npimdata1.npy /content/gdrive/'My Drive'

!cp /content/npimdata.npy /content/gdrive/'My Drive'

imdata=np.load('npimdata.npy')

imdata1=np.load('npimdata1.npy')

import csv 
from sklearn.decomposition import PCA
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import matplotlib.image as img
import PIL.Image
  
# csv file name 
filename = "ImageID.csv"
fname="Y.csv"
  
# initializing the titles and rows list 
fields = [] 
rows = [] 
imdata=None
model = models.vgg16(pretrained=True)
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier
print(*list(model.children()))
#print(imdata.shape)
  
# reading csv file 
with open(filename, 'r') as csvfile: 
  # creating a csv reader object 
  csvreader = csv.reader(csvfile) 
  
  # extracting each data row one by one 
  i=0
  j=0
  for row in csvreader: 
    try:
      i=i+1
      row=row[0].split('\\')
      #print(row)
      ty=row[0]+'/'+row[1]
      #print('/content/images/'+ty)
      image = img.imread('/content/images/'+ty)
      #print(image)
      #print(image.shape)
    except:
      j=j+1
      pass
  print(i)
  print(j)
  imdata=np.zeros(shape=(i,4096),dtype=float)
  
with open(filename, 'r') as csvfile: 
  # creating a csv reader object 
  csvreader = csv.reader(csvfile) 
  i=0
  j=0
  gft=0
  for row in csvreader: 
    try:
      print(gft)
      gft=gft+1
      image=np.zeros(shape=(1,224,224,3))
      row=row[0].split('\\')
      #print(row)
      ty=row[0]+'/'+row[1]
      #print('/content/images/'+ty)
      image[0] = img.imread('/content/images/'+ty)
      #print(ty)
      #print(1)
      image=np.rollaxis(image,3,1)
      a=torch.tensor(image,dtype=torch.float32)
      #print(torch.from_numpy(image))
      imdata[i]=model(a).detach().numpy()
      #print(imdata[i])
      #print(image.shape)
      i=i+1
    except:
      j=j+1
      pass
  print(i)
  print(j)
#imdata=np.rollaxis(imdata, 3, 1)
print(imdata.shape)

#model(torch.rand(1, 3, 224, 224)).shape

import csv 
from sklearn.decomposition import PCA
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import matplotlib.image as img
import PIL.Image
  
# csv file name 
filename = "ImageID.csv"
fname="Y.csv"
  
# initializing the titles and rows list 
fields = [] 
rows = [] 
imdata1=None
model = googlenet(pretrained=True)
#print(imdata1.shape)
  
# reading csv file 
with open(filename, 'r') as csvfile: 
  # creating a csv reader object 
  csvreader = csv.reader(csvfile) 
  
  # extracting each data row one by one 
  i=0
  j=0
  for row in csvreader: 
    try:
      i=i+1
      row=row[0].split('\\')
      #print(row)
      ty=row[0]+'/'+row[1]
      #print('/content/images/'+ty)
      image = img.imread('/content/images/'+ty)
      #print(image)
      #print(image.shape)
    except:
      j=j+1
      pass
  print(i)
  print(j)
  imdata1=np.zeros(shape=(i,1024),dtype=float)
  
with open(filename, 'r') as csvfile: 
  # creating a csv reader object 
  csvreader = csv.reader(csvfile) 
  i=0
  j=0
  gft=0
  for row in csvreader: 
    try:
      print(gft)
      gft=gft+1
      image=np.zeros(shape=(1,224,224,3))
      row=row[0].split('\\')
      #print(row)
      ty=row[0]+'/'+row[1]
      #print('/content/images/'+ty)
      image[0] = img.imread('/content/images/'+ty)
      #print(ty)
      #print(1)
      image=np.rollaxis(image,3,1)
      #print(image)
      a=torch.tensor(image,dtype=torch.float32)
      #print(torch.from_numpy(image))
      imdata1[i]=model(a).detach().numpy()
      print(imdata1[i])
      #print(image.shape)
      i=i+1
    except:
      j=j+1
      pass
  print(i)
  print(j)
#imdata=np.rollaxis(imdata, 3, 1)
print(imdata1.shape)

#model(torch.rand(1, 3, 224, 224)).shape

import csv 
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import matplotlib.image as img
import PIL.Image
import warnings
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

__all__ = ['GoogLeNet', 'googlenet']

model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

_GoogLeNetOuputs = namedtuple('GoogLeNetOuputs', ['logits', 'aux_logits2', 'aux_logits1'])


def googlenet(pretrained=False, **kwargs):
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' not in kwargs:
            kwargs['aux_logits'] = False
        if kwargs['aux_logits']:
            warnings.warn('auxiliary heads in the pretrained googlenet model are NOT pretrained, '
                          'so make sure to train them')
        original_aux_logits = kwargs['aux_logits']
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False
        model = GoogLeNet(**kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['googlenet']))
        if not original_aux_logits:
            model.aux_logits = False
            del model.aux1, model.aux2
        return model

    return GoogLeNet(**kwargs)


class GoogLeNet(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False, init_weights=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 1024
        #x = self.dropout(x)
        #x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:
            return _GoogLeNetOuputs(x, aux2, aux1)
        return x


class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = x.view(x.size(0), -1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 2048
        x = F.dropout(x, 0.7, training=self.training)
        # N x 2048
        x = self.fc2(x)
        # N x 1024

        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

import csv 
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import matplotlib.image as img
import PIL.Image
  
# csv file name 
filename = "ImageID.csv"
  
# initializing the titles and rows list 
fields = [] 
rows = [] 
imdata=None
model = models.vgg16(pretrained=True)
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier
print(*list(model.children()))

image=np.zeros(shape=(1,224,224,3))
image=np.rollaxis(image,3,1)
#print(torch.from_numpy(image))
#imdata[i]=model(torch.tensor(image,dtype=torch.double)).numpy()
a=torch.tensor(image,dtype=torch.float32)
#print(a)
model(a).detach().numpy()

!git clone --recursive https://github.com/pytorch/pytorch

s='ksbfkj/asjknfkj'
s.split("//")

!export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
!python setup.py install