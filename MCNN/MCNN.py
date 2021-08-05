import torch
import torch.utils.data.dataloader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from MoorePenrose import MoorePenrose

class MCNN(nn.Module):
    def __init__(self):
        super(MCNN, self).__init__()
        # Layer operations
        self.conv1 = nn.Conv2d(1, 15, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(15, 210, kernel_size=5, padding=1)
        self.fc2 = nn.Linear(36*210, 10, bias=False)
        
        self.op_weights = []
        self.optimizer = MoorePenrose(params=self.parameters(),C=1e-3)
        
    def arrange_input_target(self, weight, stride, inputs):
        num_w, depth_w, height_w, width_w = weight.shape
        xx = []
        
        for img_index in range(0,inputs.shape[0]):
            for row_index in range(0, inputs.shape[2]-height_w, stride):
                for col_index in range(0, inputs.shape[3]-width_w, stride):
                    local_win = torch.flatten(inputs[img_index,:,row_index:height_w + row_index,col_index:width_w + col_index])
                    xx.append(local_win)
    
        self.X = torch.stack(xx)
        Xn = (self.X-torch.mean(self.X))/torch.std(self.X, unbiased=False)
        intercept = torch.ones(Xn.shape[0],1)
        self.T = torch.cat((Xn, intercept), 1)
        #return X, T #global
    
    def arrange_weights(self, weight):
        row_x, col_x = self.X.shape
        num_w, depth_w, height_w, width_w = weight.shape
        self.W = torch.zeros([height_w*width_w*depth_w, num_w])
    
        for i in range(num_w):
            self.W[:,i] = torch.flatten(weight[i, :, :, :])
        
        self.b = torch.rand([row_x,1])
        #return W, b #global
    
    def arreange_optimal_weights(self, optimal_weights,shape):
        Fmat = optimal_weights[:-1,:]
        B = optimal_weights[-1:,:]
        Fmat = torch.transpose(Fmat, 0, 1)
        B = torch.transpose(B, 0, 1)
        self.w = torch.reshape(Fmat, shape)
        nn.Conv2d.weight = nn.Parameter(self.w)
        self.op_weights.append(nn.Conv2d.weight)
        #return w #global
    
    def TrainConvLayer(self, inputs, weight, stride):
        #X, T = arrange_input(weight, stride=1, inputs=x_ini) #global
        self.arrange_input_target(weight,1,inputs)
        #W, b = arrange_weights(X, weight) #global
        self.arrange_weights(weight)
        H_int = torch.mm(self.X, self.W) + self.b
        self.H = F.relu(H_int)
        #return H, T #global
    
    def forwardConvTraining(self, x, weight, stride):
        
        finalLayer = False
        # Training convolution
        #H,T = TrainConvLayer(x, weight, stride) #global
        shape = weight[0].shape
        self.TrainConvLayer(x, weight[0], stride)
        optimal_weights = self.optimizer.train(self.H,self.T,finalLayer)
        #w = arreange_optimal_weights(optimal_weghts) #global
        self.arreange_optimal_weights(optimal_weights,shape)
        
        # First trained layer
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,kernel_size=2)
        
        shape = weight[2].shape
        # Training convolution2
        #H,T = TrainConvLayer(x, weight, stride) #global
        self.TrainConvLayer(x, weight[2], stride)
        optimal_weights = self.optimizer.train(self.H,self.T, finalLayer)
        #w = arreange_optimal_weights(optimal_weghts) #global
        self.arreange_optimal_weights(optimal_weights,shape)
        
        # Second trained layer
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,kernel_size=2)
        
        return x
    
    def forwardLastLayer(self, x, target):
        
        # Fully connected
        finalLayer = True
        x = x.view(-1, self.num_flat_features(x))
        optimal_weights = self.optimizer.train(x,target, finalLayer)
        nn.Linear.weight = optimal_weights
        self.op_weights.append(nn.Linear.weight)
        x = self.fc2(x)

        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def forward(self, x):
        nn.Conv2d.weight = self.op_weights[0]
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,kernel_size=2)
        
        nn.Conv2d.weight = self.op_weights[1]
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,kernel_size=2)
        
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc2(x)
        
        return x