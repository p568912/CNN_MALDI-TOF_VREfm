import torch
import numpy as np
from torch.utils.data import Dataset


class trainset(Dataset):
    def __init__(self, trainFileName,labelFileName):
        data=np.load(trainFileName)    
        labels =np.loadtxt(labelFileName).astype(int)
        print(data.shape)
        print(labels.shape)
        self.mz_range=data.shape[-1]
        shuffle_data = list(zip(data, labels))
        data, labels = zip(*shuffle_data)

        self.traindata = data
        self.train_labels = labels

        print("len(self.traindata)",len(self.traindata))

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
        self.mz_range=data.shape[-1]

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
