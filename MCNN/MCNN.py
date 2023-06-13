#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from morph2d import Morph2d
import torch.nn as nn
import torch.nn.functional as F

class MCNN(nn.Module):
    """Morphological-convolutiona neural network (MCNN) class"""
    def __init__(self):
        super(MCNN, self).__init__()
        
        self.morph = Morph2d(1, 4, kernel_size=5,
                               dilation=True,
                               erosion=False,
                               convolution=False,
                               sequence= False, # ("dilation", "erosion")
                               subtraction= ("original", "erosion")) # ("original", "erosion")
        self.conv1 = nn.Conv2d(4, 20, kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(20, 5, kernel_size=5, padding="same")
        self.fc = nn.Linear(1024*5, 1, bias=False)
        self.act = nn.Softmax(dim=1)

    def forward(self, x):
        """Complete process"""
        x = self.morph(x).float()
        x = F.max_pool2d(x,kernel_size=2)
        x = F.relu(x)
        x = self.conv1(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = F.relu(x)


        x = x.view(-1, self.num_flat_features(x))

        x = self.fc(x)
        x = self.act(x)

        return x

    def forwardToHidden(self, x):
        """Process stopped at hidden layer - before fully connected -.
        This process calculate the hidden output used to estimate the weights"""
        x = self.morph(x).float()
        x = F.max_pool2d(x,kernel_size=2)
        x = self.conv1(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = F.relu(x)
        x = x.view(-1, self.num_flat_features(x))
        
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

