import torch
from torch.autograd import Variable
from random import random

class MoorePenrose(object):
    def __init__(self,params,C=1e-2,forgettingfactor=1,L =100):
        self.params=list(params)
        self.is_cuda=self.params[len(self.params)-1].is_cuda
        self.C=C
        self.L=L
        self.w=self.params[len(self.params)-1]
        self.w[0].data.fill_(0) #initialize output weight as zeros
        #self.w.data.normal_()
        # For sequential learning in OS-ELM
        self.dimInput=self.params[len(self.params)-1].data.size()[1]
        self.forgettingfactor=forgettingfactor
        self.M=Variable(torch.inverse(self.C*torch.eye(self.dimInput)),requires_grad=False, volatile=True)


    def pseudoBig(self,inputs,oneHotTarget):
        xtx = torch.mm(inputs.t(), inputs) # [ n_features * n_features ]
        dimInput=inputs.size()[1]
        I = Variable(torch.eye(dimInput),requires_grad=False, volatile=True)
        if self.L > 0.0:
            mu = torch.mean(inputs, dim=0, keepdim=True)  # [ 1 * n_features ]
            S = inputs - mu
            S = torch.mm(S.t(), S)
            self.M = Variable(torch.inverse(xtx.data + self.C * (I.data+self.L*S.data)),requires_grad=False, volatile=True)
        else:
            self.M = Variable(torch.inverse(xtx.data + self.C *I.data), requires_grad=False, volatile=True)

        w = torch.mm(self.M, inputs.t())
        w = torch.mm(w, oneHotTarget)
        self.w.data = w.t().data
        return self.w

    def pseudoSmall(self,inputs,oneHotTarget):
        xxt = torch.mm(inputs, inputs.t())
        numSamples=inputs.size()[0]
        I = Variable(torch.eye(numSamples),requires_grad=False, volatile=True)
        self.M = Variable(torch.inverse(xxt.data + self.C * I.data),requires_grad=False, volatile=True)
        w = torch.mm(inputs.t(), self.M)
        w = torch.mm(w, oneHotTarget)

        self.w.data = w.t().data
        return self.w

    def train(self,inputs,targets, finalLayer):
        if finalLayer == True:
            targets = targets.view(targets.size(0),-1)
            targets = self.oneHotVectorize(targets=targets)
        
        numSamples = inputs.size()[0]
        dimInput = inputs.size()[1]
        dimTarget = targets.size()[1]

        if numSamples > dimInput:
            wei = self.pseudoBig(inputs,targets)
        else:
            wei = self.pseudoSmall(inputs,targets)
        
        return wei

    def oneHotVectorize(self,targets):
        oneHotTarget=torch.zeros(targets.size()[0],targets.max().item()+1)

        for i in range(targets.size()[0]):
            oneHotTarget[i][targets[i].data[0]]=1

        oneHotTarget=Variable(oneHotTarget,requires_grad=False, volatile=True)

        return oneHotTarget
