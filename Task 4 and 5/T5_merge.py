import numpy as np
import os
import sys
s=700
e=1000
n='test'
vgg_feat_train = np.zeros((e-s, 30,512, 7, 7))

for i in range(s,e):
    print(i)
    vgg_feat_train[i-s]=np.load('vgg_feat/'+str(i)+'.npy')

np.save('vgg_feat_'+n+'.npy',vgg_feat_train)

