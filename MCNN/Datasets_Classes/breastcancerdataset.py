#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from skimage import transform


class BreastCancerDataset(Dataset):
    """Class to read the ISIC 2019 dataset."""

    def __init__(self, csv_file, csv_file_full, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.rnsa = pd.read_csv(csv_file).sample(frac=1) # auxiliar
        self.rnsa_full = pd.read_csv(csv_file_full) # complete
        self.root_dir = root_dir
        self.transform = transform
        self.targets = self.rnsa["cancer"]

    def __len__(self):
        return len(self.rnsa)

    def __getitem__(self, idx):
        img = []
        l_cc = []
        l_mlo = []
        r_cc = []
        r_mlo = []
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_id = self.rnsa["patient_id"].iloc[idx]
        self.rnsa = self.rnsa.sample(frac=1)
        folder_name = os.path.join(self.root_dir,
                                   str(patient_id))
        
        for images in os.listdir(folder_name):
            img_path = os.path.join(folder_name, images)
            image = io.imread(img_path)
            img.append(image)
        
        label_per_patient = self.rnsa["cancer"].iloc[idx]
        label_per_image = list(self.rnsa_full[self.rnsa_full["patient_id"]==patient_id]["cancer"])
        l_cc.append(label_per_image[0])
        l_mlo.append(label_per_image[1])
        r_cc.append(label_per_image[2])
        r_mlo.append(label_per_image[3])

        sample = {'image': img,
                  'label_patient': label_per_patient,
                  'label_l_cc': l_cc,
                  'label_l_mlo': l_mlo,
                  'label_r_cc': r_cc,
                  'label_r_mlo': r_mlo}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        list_img, label_patient = sample['image'], sample['label_patient']
        l_cc, l_mlo, r_cc, r_mlo = sample['label_l_cc'], sample['label_l_mlo'], sample['label_r_cc'], sample['label_r_mlo']
        new_list_img = []

        for image in list_img:
            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size, self.output_size * w / h
                else:
                    new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            img = transform.resize(image, (new_h, new_w))

            if new_h > new_w:
                required_pad1 = (new_h - new_w)/2

                if required_pad1%2 != 0 and required_pad1%2 != 1:
                  required_pad1 += 0.5
                  required_pad2 = int(required_pad1 - 1)
                else:
                  required_pad2 = int(required_pad1)
            
                required_pad1 = int(required_pad1)
                img = np.pad(img, ((0, 0),(required_pad1, required_pad2), (0, 0)), 'constant', constant_values=(0))
            elif new_w > new_h:
                required_pad1 = (new_w - new_h)/2
                if required_pad1%2 != 0 and required_pad1%2 != 1:
                  required_pad1 += 0.5
                  required_pad2 = int(required_pad1 - 1)
                else:
                  required_pad2 = int(required_pad1)
            
                required_pad1 = int(required_pad1)
                img = np.pad(img, ((required_pad1, required_pad2),(0, 0), (0, 0)), 'constant', constant_values=(0))

            new_list_img.append(img)

        return {'image': new_list_img,
                'label_patient': label_patient,
                'label_l_cc': l_cc,
                'label_l_mlo': l_mlo,
                'label_r_cc': r_cc,
                'label_r_mlo': r_mlo}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label_patient = sample['image'], sample['label_patient']
        l_cc, l_mlo, r_cc, r_mlo = sample['label_l_cc'], sample['label_l_mlo'], sample['label_r_cc'], sample['label_r_mlo']
        ## Crear numpy array de H x W x C=4
        image = np.array(image)
        image = image[:,:,:,0]

        ##image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label_patient': torch.Tensor([label_patient]),
                'label_l_cc': torch.Tensor(l_cc),
                'label_l_mlo': torch.Tensor(l_mlo),
                'label_r_cc': torch.Tensor(r_cc),
                'label_r_mlo': torch.Tensor(r_mlo)} 

