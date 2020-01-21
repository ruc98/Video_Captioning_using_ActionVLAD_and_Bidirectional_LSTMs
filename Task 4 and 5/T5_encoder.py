import numpy as np
import os
import cv2
import torch
from torch import nn
from torchvision import transforms
import torchvision.models as models
from torchsummary import summary
import googlenet as gnetmodel
torch.cuda.set_device(0)
vgg16model = models.vgg16(pretrained=True)
vgg16_conv = nn.Sequential(*list(vgg16model.children())[:-1])

goognet = gnetmodel.googlenet(pretrained=True)
gnet_conv = nn.Sequential(*list(goognet.children())[:-3])

vgg16_conv.cuda()
gnet_conv.cuda()

trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((224,224)), 
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

video_folder  = 'YouTubeClips/'
vid_count=0
files = 'video_data/videoIDs.txt'
with open(files) as f:
    s=f.readlines() 
    for idx,i in enumerate(s):
        j = i[:-1]
        vid_path = video_folder+str(j)+'.avi'

        cap = cv2.VideoCapture(vid_path)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret = True
        fc = 0
        c=0
        framestep = int(frameCount/30)
        videos = torch.zeros([30,3,224,224]).cuda()
        while (fc < frameCount  and ret):
            ret, frame = cap.read()
            fc+=1
            
            if fc % framestep==0:
                if c>=30:
                    break
                newframe = trans(frame)
                videos[c] = newframe.cuda()
                c+=1
        featvgg = vgg16_conv(videos)
        featgnet= gnet_conv(videos)
        featvgg = featvgg.cpu()
        featgnet = featgnet.cpu()
        np.save('vgg_feat/'+str(vid_count)+'.npy',featvgg.detach().numpy())
        np.save('gnet_feat/'+str(vid_count)+'.npy',featgnet.detach().numpy())

        vid_count+=1        

print('Done!!')
