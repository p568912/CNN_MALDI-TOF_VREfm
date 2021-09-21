import numpy as np
import random
import time
import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import csv
from sklearn.metrics import roc_auc_score
import copy

import cv2
import gc
import heapq
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


class trainset(Dataset):
    def __init__(self, trainFileName,labelFileName):
        data=np.load(trainFileName)    
        labels =np.loadtxt(labelFileName).astype(int)
        print(data.shape)
        print(labels.shape)
        shuffle_data = list(zip(data, labels))
        data, labels = zip(*shuffle_data)

        self.traindata = data
        self.train_labels = labels

    def __getitem__(self, index):
        data = self.traindata[index]
        label = self.train_labels[index]
        # revcomp = self.revcomp_transform(data)
        #data = np.expand_dims(data, axis=0)
        # revcomp = np.expand_dims(revcomp, axis=0)
        # data = np.concatenate((data, revcomp), axis=0)

        return data, label

    def __len__(self):
        return len(self.traindata)

class testset(Dataset):
    def __init__(self, testFileName,labelFileName):

        data=np.load(testFileName)
        labels =np.loadtxt(labelFileName).astype(int)

        print(data.shape)
        print(labels.shape)
        """
        shuffle_data = list(zip(data, labels))
        random.shuffle(shuffle_data)
        data, labels = zip(*shuffle_data)
        """
        self.testdata = data
        self.test_labels = labels

    def __getitem__(self, index):
        data = self.testdata[index]
        label = self.test_labels[index]
        # revcomp = self.revcomp_transform(data)
        data = np.expand_dims(data, axis=0)
        # revcomp = np.expand_dims(revcomp, axis=0)
        # data = np.concatenate((data, revcomp), axis=0)
        return data, label

    def __len__(self):
        return len(self.testdata)

class baseCNN3(nn.Module):  # Inherit from `nn.Module`, define `__init__` & `forward`
    def __init__(self,channels,ReLUFlag,poolingFlag):
        # Always call the init function of the parent class `nn.Module`
        # so that magics can be set up.
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=5)
        self.conv4 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=5)
        
        self.globalpooling = nn.AdaptiveMaxPool1d(1)

        self.poolingFlag=poolingFlag
        self.ReLUFlag=ReLUFlag

        self.drop1=nn.Dropout(p=0.5)
        self.drop2=nn.Dropout(p=0.5)

        max_pooling=nn.MaxPool1d(3, stride=2)
        pooling=nn.AvgPool1d(5,stride=2,padding=5//2)
        self.pooling=nn.AvgPool1d(5,stride=2,padding=5//2)

        self.in1=nn.InstanceNorm1d(9000)
        self.bn1=nn.BatchNorm1d(channels)
        self.bn2=nn.BatchNorm1d(channels)
        self.bn3=nn.BatchNorm1d(channels)
        self.bn4=nn.BatchNorm1d(channels)
        
        if ReLUFlag:
            self.conv_layers = nn.Sequential(
                #self.in1,
                self.conv1,
                self.bn1,
                nn.ReLU(),
                #nn.Tanh(),
                pooling,
                self.conv2,
                self.bn2,
                nn.ReLU(),
                #nn.Tanh(),
                pooling,
                self.conv3,
                self.bn3,
                nn.ReLU(),
                #nn.Tanh(),
                pooling,
            )
        else:
            self.conv_layers = nn.Sequential(
                #self.in1,
                self.conv1,
                self.bn1,
                nn.Tanh(),
                pooling,
                self.conv2,
                self.bn2,
                nn.Tanh(),
                pooling,
                self.conv3,
                self.bn3,
                nn.Tanh(),
                pooling,
            )

        
        x = torch.randn(1,18000).view(-1,1,18000)
        self._to_linear = None
        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear, 64) #flattening.
        self.classifier = nn.Sequential(
			#nn.Linear(256, 32),
            nn.Linear(64, 1),
			nn.Sigmoid(),
		)
        

        # Define the parameters in your network.
        # This is achieved by defining the shapes of the multiple layers in the network.
    def convs(self,x):
        
        # Define two 2D convolutional layers (1 x 10, 10 x 20 each)
        # with convolution kernel of size (5 x 5).
        # Define a dropout layer
        if self.poolingFlag == True:
            x=self.pooling(x)
        x=self.conv_layers(x)
       
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]
        return x

     
        
        
    def forward(self, x):
        # Define the network architecture.
        # This is achieved by defining how the network forward propagates your inputs
        # input 64 * 101 *4
        # Input image size: 28 x 28, input channel: 1, batch size (training): 64 
        #print(x.size())
        # Input (64 x 4 x 101) -> Conv1 (64 x 16 x 78) -> Max Pooling (64 x 16 x 26) -> ReLU -> ...
        x = self.convs(x)
        #print(x.size())
        #dimension 1123
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        #print(x.size())
        x=self.drop1(x)
        x = F.relu(self.fc1(x))
        x=self.drop2(x)
        #print(x.size())
        #x = self.fc2(x) # bc this is our output layer. No activation here.
        x = self.classifier(x)
        
        #print(x.size())
        
        return x

def train(model, train_loader, optimizer, criterion, epoch):
    print("Epoch {:d}".format(epoch))
    model.train()
    correct = 0
    all_label = []
    all_pred = []
    for (data, target) in train_loader:
        data=data.view(-1, 1,18000).float()
        #print(data.size())
        #target = target.long()
        
        data, target = data.cuda(), target.cuda()
        target = target.float()
        optimizer.zero_grad()
       
        output = model(data)
        
        pred = output>0.5
        correct += pred.eq(target.view_as(pred)).sum().item()
        all_label.extend(target.reshape(-1).tolist())
        all_pred.extend((output[:]).reshape(-1).tolist())

        output = output.reshape(-1)
        loss = criterion(output, target)
        """
        print("output :{}".format(output))
        print("target :{}".format(target))
        print("loss: {}".format(loss))
        """
        """
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        all_label.extend(target.reshape(-1).tolist())
        all_pred.extend((output[:,1]).reshape(-1).tolist())
        """
        #target_onehot = torch.FloatTensor(output.shape)
        
        loss = criterion(output, target) #######-0.1*dice_loss(output, target_onehot) 
        loss.backward()
        optimizer.step()

    # print("Epoch %d\nTraining acc: %.2f"%(epoch, 100. * correct/len(train_loader.dataset))+"%")
    print("Train AUC score: {:.4f}".format(roc_auc_score(np.array(all_label), np.array(all_pred))))

def test(model, test_loader, criterion,predPath, data_type='Test', arch=None):
    model.eval()
    correct = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    all_label = []
    all_pred = []
    for batch_idx, (data, target) in enumerate(test_loader):
        #target = target.long()
        data=data.view(-1, 1,18000).float()
        data, target = data.cuda(), target.cuda()
        target = target.float()

        #output = model(data, arch=None)
        output = model(data)
        """
        output = F.softmax(output, dim=1)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        """
        pred = output>0.5
        correct += pred.eq(target.view_as(pred)).sum().item()
        for p, t in zip(pred, target.view_as(pred)):
            if p.eq(t) and p.item()==1:
                tp += 1
            elif  p.eq(t) and p.item()==0:
                tn += 1
            elif p.item()==1:
                fp += 1
            else:
                fn += 1
        all_label.extend(target.reshape(-1).tolist())
        all_pred.extend((output[:]).reshape(-1).tolist())
        #all_label.extend(target.reshape(-1).tolist())
        #all_pred.extend((output[:,1]).reshape(-1).tolist())
        
        
    # print('Testing acc: {:.2f}% ({}/{})'.format(100. * correct / len(test_loader.dataset), correct, len(test_loader.dataset)))
    accuracy=0
    Specificity=0
    Sensitivity=0
    if data_type=='Test':
        accuracy=100. * (tp+tn) / len(all_label)
        Specificity=100. * tn / (tn+fp)
        Sensitivity=100. * tp / (tp+fn)
        print("false negatives: {} ({:.2f}%)".format(fn, 100. * fn / len(all_label)))
        print("false positives: {} ({:.2f}%)".format(fp, 100. * fp / len(all_label)))
        print("true positives: {} ({:.2f}%)".format(tp, 100. * tp / len(all_label)))
        print("true negatives: {} ({:.2f}%) \n".format(tn, 100. * tn / len(all_label)))
        print("accuracy:      ({:.2f}%)".format( accuracy))
        print("Specificity:   ({:.2f}%)".format( Specificity))
        print("Sensitivity:   ({:.2f}%)".format( Sensitivity))
        

        pred_res = np.concatenate((np.array(all_label), np.array(all_pred)),axis=0).reshape(2,-1).T
        np.savetxt(predPath, pred_res, delimiter=",",fmt='%.6f')

    
    print("{} AUC score: {:.4f}".format(data_type, roc_auc_score(np.array(all_label), np.array(all_pred))))
    return roc_auc_score(np.array(all_label), np.array(all_pred)),accuracy,Specificity,Sensitivity

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    """
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        tf.random.set_seed(seed)

    """
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',help='seed number',type=int,default=0)
    parser.add_argument('--poolingFlag',default=True,type=str2bool)
    parser.add_argument('--ReLUFlag',default=True,type=str2bool)
    parser.add_argument('--channels',default=64,type=int)
    parser.add_argument('--savePath', required=True)
    parser.add_argument('--predPath', required=True)

    args = parser.parse_args()
    print(args)

    seed_num=args.seed
    set_seed(seed_num)
    
    train_data = trainset('Linkou_EF_Data_round_off_dim_18000.npy','Linkou_EF_Data_labels.csv')
    test_data = testset('Kaohsiung_EF_Data_round_off_dim_18000.npy','Kaohsiung_EF_Data_labels.csv')
    
    dataset_size = len(train_data)
    print(dataset_size)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)


    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    batch_size = 32
    epochs = 30
    learning_rate =0.0001

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, pin_memory=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=2)

    model = baseCNN3(channels=args.channels,ReLUFlag=args.ReLUFlag,poolingFlag=args.poolingFlag)
    #model=baseCNN_NIN()
    #model = InceptionModule2()
    #model = baseCNN_skipConnection()
    model = model.cuda()

    print("model size: %d"%sum(p.numel() for p in model.parameters()))
    print(model)

    #criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 1.0]))
    #criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 1.0]).cuda())
    criterion = nn.BCELoss()
    criterion = criterion.cuda()
    #optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #epochs=1000

    best_auc=0
    best_epoch=0

    model_check_epoch = copy.deepcopy(model)
    #optimizer_check_epoch = torch.optim.SGD(model_check_epoch.parameters(), lr= learning_rate)
    optimizer_check_epoch = torch.optim.Adam(model_check_epoch.parameters(), lr= learning_rate)
    criterion_check = nn.BCELoss()
    criterion_check = criterion_check.cuda()

    for epoch in range(1, epochs):
        train(model_check_epoch, train_loader, optimizer_check_epoch, criterion_check, epoch)
        auc,accuracy,Specificity,Sensitivity=test(model_check_epoch, valid_loader, criterion_check,args.predPath, data_type='valid')
        if auc > best_auc:
            best_auc=auc
            best_epoch=epoch

    print("best_epoch",best_epoch)
    print("valid_auc:",best_auc)
    if best_epoch==1:
        best_epoch=2
    for epoch in range(1, best_epoch):
        train(model, train_loader, optimizer, criterion, epoch)
        train(model, valid_loader, optimizer, criterion, epoch)
        auc,accuracy,Specificity,Sensitivity=test(model, test_loader, criterion,args.predPath)

    print(auc)


    torch.save(model,args.savePath)


    return auc,accuracy,Specificity,Sensitivity

if __name__ == '__main__':
    start=datetime.now()
    auc,accuracy,Specificity,Sensitivity=main()
    end=datetime.now()
    print("total seconds:",end-start)
    print("{} {} {} {}".format(auc,accuracy,Specificity,Sensitivity))
   