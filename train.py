#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader


__all__ = ['vgg16_bn']
model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


class VGG(nn.Module):

    def __init__(self, features, output_size=1274, image_size=448):
        super(VGG, self).__init__()
        self.features = features
        self.image_size = image_size

        self.yolo = nn.Sequential(
            nn.Linear(49*512,4096),
            nn.Linear(4096,1274)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.yolo(x)
        x = torch.sigmoid(x) 
        x = x.view(-1,7,7,26)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    s = 1
    first_flag=True
    for v in cfg:
        s=1
        if (v==64 and first_flag):
            s=2
            first_flag=False
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=s, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def conv_bn_relu(in_channels,out_channels,kernel_size=3,stride=2,padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}




def Yolov1_vgg16bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    yolo = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        vgg_state_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        yolo_state_dict = yolo.state_dict()
        for k in vgg_state_dict.keys():
            if k in yolo_state_dict.keys() and k.startswith('features'):
                yolo_state_dict[k] = vgg_state_dict[k]
    yolo.load_state_dict(yolo_state_dict)
    return yolo



def test():
    import torch
    model = Yolov1_vgg16bn(pretrained=True)
    img = torch.rand(1,3,448,448)
    output = model(img)
    print(output.size())


# In[2]:


import hw2_evaluation_task as t


# In[3]:


import numpy as np
import cv2
#import torchvision
#import torchvision.transforms as transforms
import torch.optim as optim


# In[37]:


def get_w(bbox):
    return (bbox[2]-bbox[0])/448
def get_h(bbox):
    return (bbox[3]-bbox[1])/448
def get_center(bbox):
    return [(bbox[2]+bbox[0])/2,(bbox[3]+bbox[1])/2]
def get_xy(g,center):
    return [(center[0]-g%7*64)/64,(center[1]-int(g/7)*64)/64]
def to_cat(c):
    cat = np.zeros(16)
    e = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle'
    , 'ship', 'tennis-court','basketball-court', 'storage-tank',  'soccer-ball-field','roundabout'
    , 'harbor', 'swimming-pool', 'helicopter', 'container-crane'].index(c)
    cat[e] = 1
    return cat
def to_label(gts):
    GTS=[]
    for gt in gts:
        cat = to_cat(gt['name'])
        d = gt['difficult']
        bbox = gt['bbox']
        GTS.append(np.concatenate((bbox,[d],cat)))
    return GTS
def get_grid(bbox):
    return int((bbox[0]+bbox[2])/2/64)+int((bbox[1]+bbox[3])/2/64)*7
def to_xywh(bbox):#(xyxy--------)
    g = get_grid(bbox)
    c = get_center(bbox)
    return np.concatenate((get_xy(g,c),[get_w(bbox),get_h(bbox)],bbox[4:]))
def calculate_iou(bbox_pred, bbox):
    left = max(bbox_pred[0],bbox[0])
    right = min(bbox_pred[2],bbox[2])
    top = max(bbox_pred[1],bbox[1])
    bot = min(bbox_pred[3],bbox[3])
    print([left,right,top,bot])
    if (left<right) & (top<bot):
        i = calculate_area([left,top,right,bot])
        s = calculate_area(bbox_pred)+calculate_area(bbox)
        print([i,s])
        return i/(s-i)
    else: return 0
def calculate_iou_xywh(bbox_pred, bbox):
    batch_size = bbox_pred.size(0)
    p_x = bbox_pred[:,:,:,0]*64
    gt_x = bbox[:,:,:,0]*64
    p_y = bbox_pred[:,:,:,1]*64
    gt_y = bbox[:,:,:,1]*64
    p_w = bbox_pred[:,:,:,2]*448/2
    gt_w = bbox[:,:,:,2]*448/2
    p_h = bbox_pred[:,:,:,3]*448/2
    gt_h = bbox[:,:,:,3]*448/2
    left = torch.max((p_x-p_w),(gt_x-gt_w)).unsqueeze(3)
    right = torch.min((p_x+p_w),(gt_x+gt_w)).unsqueeze(3)
    top = torch.max((p_y-p_h),(gt_y-gt_h)).unsqueeze(3)
    bot = torch.min((p_y+p_h),(gt_y+gt_h)).unsqueeze(3)
    zeros = torch.zeros(batch_size, 7, 7, 1)
    i = torch.max(right-left,zeros)*torch.max(right-left,zeros)
    s = p_w*p_h+gt_w*gt_h
    return i/(s.unsqueeze(3)-i) #[:,:,:,1]


# In[5]:


def get_img_train(s,n):
    X_train=np.zeros((n,448,448,3),np.float32)
    for i in range(s,s+n):
        #print('hw2_train_val/train15000/images/'+str('%05d'%i)+'.jpg')
        X_train[i-s]=(cv2.imread('hw2_train_val/train15000/images/'+str('%05d'%i)+'.jpg')[:448,:448,:])/255
    return X_train
def get_img_test(s,n):
    X_train=np.zeros((n,448,448,3),np.float32)
    for i in range(s,s+n):
        #print('hw2_train_val/train15000/images/'+str('%05d'%i)+'.jpg')
        X_train[i-s]=(cv2.imread('hw2_train_val/val1500/images/'+str('%04d'%i)+'.jpg')[:448,:448,:])/255
    return X_train
def get_label_train(s,n):
    y_train=np.zeros((n,7,7,22),np.float32)
    for i in range(s,s+n):
        #print('hw2_train_val/train15000/images/'+str('%05d'%i)+'.jpg')
        gts = t.parse_gt('hw2_train_val/train15000/labelTxt_hbb/'+str('%05d'%i)+'.txt')
        y = to_label(gts)
        for b in y:
            g = get_grid(b)
            if g>=49:
                continue
            y_train[i-s][int(g/7)][g%7] = np.concatenate((to_xywh(b),[1]))
    return y_train
def get_label_test(s,n):
    y_train=np.zeros((n,7,7,22),np.float32)
    for i in range(s,s+n):
        #print('hw2_train_val/train15000/images/'+str('%05d'%i)+'.jpg')
        gts = t.parse_gt('hw2_train_val/val1500/labelTxt_hbb/'+str('%04d'%i)+'.txt')
        y = to_label(gts)
        for b in y:
            g = get_grid(b)
            if g>=49:
                continue
            y_train[i-s][int(g/7)][g%7] = np.concatenate((to_xywh(b),[1]))
    return y_train


# In[6]:


def cal_loss(y_pred,y_label): #(1,7,7,26),(1,7,7,22)
    b1 = y_pred[:,:,:,:5]
    b2 = y_pred[:,:,:,5:10]
    bbox = y_label[:,:,:,:5]
    c = y_pred[:,:,:,10:26]-y_label[:,:,:,5:21]
    iou_1 = calculate_iou_xywh(b1, bbox)
    iou_2 = calculate_iou_xywh(b2, bbox)
    iou = torch.cat((iou_1,iou_2),3)
    maxarg = torch.max(iou,3)[1].unsqueeze(3).float()
    bp = torch.mul(b1,maxarg)+torch.mul(b2,torch.ones_like(maxarg)-maxarg)
    loss=5*torch.sum((torch.pow((bp[:,:,:,0]-bbox[:,:,:,0]),2)+torch.pow((bp[:,:,:,1]-bbox[:,:,:,1]),2))*y_label[:,:,:,21],(1,2))
    +5*torch.sum(((bp[:,:,:,2]**0.5-bbox[:,:,:,2]**0.5)**2+(bp[:,:,:,3]**0.5-bbox[:,:,:,3]**0.5)**2)*y_label[:,:,:,21],(1,2))
    +1*torch.sum(torch.sum(torch.pow(c,2),(3))*y_label[:,:,:,21],(1,2))
    +0.5*torch.sum(((y_pred[:,:,:,4])**2+(y_pred[:,:,:,9])**2)*(torch.ones_like(y_label[:,:,:,21])-y_label[:,:,:,21]),(1,2))
    return loss


# In[7]:


# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)


# In[8]:


x_train = get_img_train(0,500)
x_test = get_img_test(0,500)


# In[9]:


y_train = get_label_train(0,500)
y_test = get_label_test(0,500)


# In[10]:


trainset = torch.utils.data.TensorDataset(torch.from_numpy(x_train.reshape(500,3,448,448)),torch.from_numpy(y_train))
testset = torch.utils.data.TensorDataset(torch.from_numpy(x_test.reshape(500,3,448,448)),torch.from_numpy(y_test))


# In[11]:


model = Yolov1_vgg16bn(pretrained=True)
model = model.float()


# In[12]:


trainset_loader = DataLoader(trainset, batch_size=20, shuffle=True, num_workers=1)
testset_loader = DataLoader(testset, batch_size=300, shuffle=False, num_workers=1)


# In[13]:


dataiter = iter(trainset_loader)
images, labels = dataiter.next()


# In[41]:


def train(model, epoch, log_interval=2):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()  # Important: set training mode
    iteration = 0
    for ep in range(epoch):
        print('ep='+str(ep))
        for batch_idx, (data, target) in enumerate(trainset_loader):
            print('batch_idx='+str(batch_idx))
            #data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            y_pred = output
            loss = cal_loss(y_pred, target)
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader)))
                print(loss)
            iteration += 1


# In[42]:


train(model, 5)


# In[19]:


p = torch.randn(10,7,7,26)
g = torch.randn(10,7,7,26)


# In[36]:


cal_loss(p,g)


# In[28]:


t = torch.randn(10,7,7)


# In[30]:


t.unsqueeze(3).shape


# In[ ]:




