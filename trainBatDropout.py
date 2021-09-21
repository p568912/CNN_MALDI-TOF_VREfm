import numpy as np
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
from utils import set_seed
from utils import str2bool
from utils import choose_optimizer
from dataset import trainset
from dataset import testset
from scorecam import scorecam

class baseCNN(nn.Module):  # Inherit from `nn.Module`, define `__init__` & `forward`
    def __init__(self,args,device,mz_range=18000,num_neuron=64,kernel_size=5,drop_p1=0.5,drop_p2=0.5,poolingSize=5):
        # Always call the init function of the parent class `nn.Module`
        # so that magics can be set up.
        super().__init__()
        self.device=device
        channels=args.channels
        ReLUFlag=args.ReLUFlag
        poolingFlag=args.poolingFlag
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size)
        self.conv3 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size)
        self.conv4 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size)
        
        self.globalpooling = nn.AdaptiveMaxPool1d(1)

        self.poolingFlag=poolingFlag
        self.ReLUFlag=ReLUFlag

        self.drop1=nn.Dropout(p=drop_p1)
        self.drop2=nn.Dropout(p=drop_p2)

        pooling=nn.AvgPool1d(poolingSize,stride=2,padding=poolingSize//2)
        self.pooling=nn.AvgPool1d(poolingSize,stride=2,padding=poolingSize//2)

        
        if ReLUFlag:
            activation=nn.ReLU()
        else:
            activation=nn.Tanh()

        self.conv_layers = nn.Sequential(
            #self.in1,
            self.conv1,
            nn.BatchNorm1d(channels),
            activation,
            pooling,
            self.conv2,
            nn.BatchNorm1d(channels),
            activation,
            pooling,
            self.conv3,
            nn.BatchNorm1d(channels),
            activation,
            pooling,
        )

        x = torch.randn(1,mz_range).view(-1,1,mz_range)
        self._to_linear = None
        self.convs(x)
        

        self.fc1 = nn.Linear(self._to_linear, num_neuron) #flattening.
        self.classifier = nn.Sequential(
			#nn.Linear(256, 32),
            nn.Linear(num_neuron, 1),
			nn.Sigmoid(),
		)
        

    # This is achieved by defining the shapes of the multiple layers in the network.
    def convs(self,x):
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
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x=self.drop1(x)
        x = F.relu(self.fc1(x))
        #x=self.fc1(x)
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
        data=data.view(-1, 1,data.shape[-1]).float()
        
        data, target = data.to(model.device), target.to(model.device)
        target = target.float()
        optimizer.zero_grad()
       
        output = model(data)
        
        pred = output>0.5
        correct += pred.eq(target.view_as(pred)).sum().item()
        all_label.extend(target.reshape(-1).tolist())
        all_pred.extend((output[:]).reshape(-1).tolist())

        output = output.reshape(-1)
        loss = criterion(output, target)
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
        data=data.view(-1, 1,data.shape[-1]).float()
        data, target = data.to(model.device), target.to(model.device)
        target = target.float()

        output = model(data)

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



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',help='seed number',type=int,default=0)
    parser.add_argument('--poolingFlag',default=True,type=str2bool)
    parser.add_argument('--ReLUFlag',default=True,type=str2bool)
    parser.add_argument('--showPosImportance',default=True,type=str2bool)
    parser.add_argument('--channels',default=64,type=int)
    parser.add_argument('--savePath', required=True)
    parser.add_argument('--predPath', required=True)

    parser.add_argument('--trainData', default='/volume/tsungting/MALDI-TOF/MALDI-TOF/20210414/Linkou_EF_Data_round_off_dim_18000.npy', type=str)
    parser.add_argument('--trainLabel', default='/volume/tsungting/MALDI-TOF/MALDI-TOF/20210414/Linkou_EF_Data_labels.csv', type=str)
    parser.add_argument('--testData', default='/volume/tsungting/MALDI-TOF/MALDI-TOF/20210414/Kaohsiung_EF_Data_round_off_dim_18000.npy', type=str)
    parser.add_argument('--testLabel', default='/volume/tsungting/MALDI-TOF/MALDI-TOF/20210414/Kaohsiung_EF_Data_labels.csv', type=str)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'adagrad'])
    parser.add_argument('--epochs', type=int, default=30, help='num of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='init learning rate')
    parser.add_argument('--splitRatio', type=float, default=0.2, help='init learning rate')

    args = parser.parse_args()
    print(args)

    seed_num=args.seed
    set_seed(seed_num)
    
    train_data = trainset(args.trainData,args.trainLabel)
    test_data = testset(args.testData,args.testLabel)

    dataset_size = len(train_data)
    print(dataset_size)
    indices = list(range(dataset_size))
    split = int(np.floor(args.splitRatio * dataset_size))
    np.random.shuffle(indices)


    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, pin_memory=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=2)

    device = 'cpu' if args.cuda==-1 else 'cuda:%d'%args.cuda
    model = baseCNN(args,device=device)

    model = model.to(device)

    print(model)
    print("model size: %d"%sum(p.numel() for p in model.parameters()))


    criterion = nn.BCELoss()
    criterion = criterion.to(device)
    #optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)
    optimizer = choose_optimizer(args.optimizer, model, learning_rate)
    #epochs=1000

    best_auc=0
    best_epoch=0

    model_check_epoch = copy.deepcopy(model)
    #optimizer_check_epoch = torch.optim.SGD(model_check_epoch.parameters(), lr= learning_rate)
    optimizer_check_epoch = choose_optimizer(args.optimizer, model_check_epoch, learning_rate)
    criterion_check = nn.BCELoss()
    criterion_check = criterion_check.to(device)

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

    if args.showPosImportance:
        out_mask=scorecam(model,model.pooling,test_data)
        np.save('model_avgpool_seed_{}_score_cam.npy'.format(seed_num), out_mask)


    return auc,accuracy,Specificity,Sensitivity

if __name__ == '__main__':
    start=datetime.now()
    auc,accuracy,Specificity,Sensitivity=main()
    end=datetime.now()
    print("total seconds:",end-start)
    print("{} {} {} {}".format(auc,accuracy,Specificity,Sensitivity))
   
