import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm.auto import tqdm
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffusionDataset(Dataset):
    def __init__(self,file,transform=None):
        self.file_object=h5py.File(file,"r")
        self.feature=self.file_object.get('feature')
        self.label=self.file_object.get('label')
        self.transform = transform
        pass
    def __len__(self):
        return len(self.feature)
    def __getitem__(self, index):
        if (index >= len(self.feature)):
            raise IndexError()
        feature = np.array(self.feature[index],dtype=np.int32)
        label = np.array(self.label[index],dtype=np.float32)
        feature=torch.Tensor(feature).type(torch.float32).unsqueeze(0)
        mask=torch.Tensor(label).type(torch.float32)
        if self.transform is not None:
            feature = self.transform(feature)
            feature=feature.repeat(3,1,1)
        return (feature,mask)
    def plot_image(self,index):
        fig,(ax)=plt.subplots(nrows=1,ncols=1)
        ax.imshow(np.array(self.feature[index]),cmap='bone')
        plt.show()
        pass
    pass

if __name__=='__main__':
    diffusiondataset = DiffusionDataset("../ViT_Dataset/Test/Dataset_Test.h5")
    plt.plot(diffusiondataset[0][1][:,0],diffusiondataset[0][1][:,1])
    plt.show()