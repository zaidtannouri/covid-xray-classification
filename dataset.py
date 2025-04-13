# dataset.py
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch

class CovidDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, -1]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


class CovidDatasetDual(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        img_path1 = self.dataframe.iloc[idx, 0]
        img_path2 = self.dataframe.iloc[idx, 1]
        label = self.dataframe.iloc[idx, -1]

        image1 = Image.open(img_path1).convert("RGB")
        image2 = Image.open(img_path2).convert("RGB")
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        label = torch.tensor(label, dtype=torch.long)
        return image1, image2, label








