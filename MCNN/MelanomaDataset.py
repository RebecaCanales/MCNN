from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from google.colab import files
import importlib_metadata
from skimage import io, transform, color

##uploaded = files.upload()

##melanoma_annotations = pd.read_csv(io.BytesIO(uploaded['.csv']))
##print(melanoma_annotations.iloc[0, -1])
##melanoma_annotations.head()

class MelanomaDataset(Dataset):
    """Melanoma dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.read_csv(io.BytesIO(uploaded['.csv'])) # when used in colab
        self.annotations = pd.read_csv('csv_file') # when not used in colab
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.annotations.iloc[idx, 0])
        image = io.imread(img_name)
        labels = self.annotations.iloc[idx, -1]
        labels = np.array([labels])
        labels = labels.astype('float').reshape(-1, 2)
        sample = {'image': image, 'label': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample