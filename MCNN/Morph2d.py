'''
This section contains the class for the morphological layer: Morph2d
'''

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Morph2d(nn.Module):
    """ Custom Dilation layer. Only for square kernel """
    # Morph2d(1, 16, kernel_size=5, padding=1)  [done] CAMBIAR DE 15 A 16 EN EL MÉTODO PRINCIPAL
    def __init__(self, in_channels, 
                 out_channels, 
                 kernel_size, 
                 dilation=True, 
                 erosion=True, 
                 opening=True, 
                 closing=True,
                 subtraction=("dilation", "erosion"),
                 stride=1, 
                 padding=0):
        super().__init__()

        """
        Creating a tensor with operations information.
        dilation = 1
        erosion = 2
        opening  = 3
        closing = 4
        """

        op_num = []
        if dilation == True:
            op_num.append(1)
        if erosion == True:
            op_num.append(2)
        if opening == True:
            op_num.append(3)
        if closing == True:
            op_num.append(4)
        if subtraction != False:
            op_num.append(5)
        self.operations = torch.Tensor(op_num)

        self.subtraction = subtraction
        self.size_in, self.size_out = in_channels, out_channels
        weight = torch.Tensor(int(out_channels/len(self.operations)), in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(weight)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(out_channels)
        self.bias = nn.Parameter(bias)
        self.padding = padding
        self.stride = stride        

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
        w = torch.nn.functional.relu(self.weight)
        w[w>0] = 1
        self.weight = nn.Parameter(w)
        print("Binarized weights: ", self.weight)
        
        
    def forward(self, x):
        print("Start of dilation layer")
        print("Weights for inspection in morph: ", self.weight)
        pad_parameter = (self.padding, self.padding, self.padding, self.padding)
        out_height = int(((x.shape[2] + 2*self.padding - (self.weight.shape[2] - 1) - 1)/self.stride) + 1)
        out_width = int(((x.shape[3] + 2*self.padding - (self.weight.shape[3] - 1) - 1)/self.stride) + 1)
        
        filter_size = self.weight.shape[2]
        result = torch.zeros(x.shape[0], self.size_out, out_height, out_width) # dimensions of image are [batch, output channels, height, width]

        img_dilation = torch.zeros(out_height, out_width)
        img_erosion = torch.zeros(out_height, out_width)
        img_opening = torch.zeros(out_height, out_width)
        img_closing = torch.zeros(out_height, out_width)

        img_dilation = F.pad(img_dilation, pad_parameter, "constant", 0)
        img_erosion = F.pad(img_erosion, pad_parameter, "constant", 0)

        img_dilation = F.pad(img_dilation, pad_parameter, "constant", 0)
        img_erosion = F.pad(img_erosion, pad_parameter, "constant", 0)

        x_padded = F.pad(x, pad_parameter, "constant", 0)

        def dilation_ordinary(curr_region, n, num_weight, in_c, r, c):
            #Element-wise addition between the current region and the filter.
            curr_result = abs(curr_region * self.weight[num_weight, in_c, :, :])
            dilation_result = torch.max(curr_result)
            return dilation_result

        def erosion_ordinary(curr_region, n, num_weight, in_c, r, c):
            #Element-wise addition between the current region and the filter.
            curr_result = abs(curr_region + self.weight[num_weight, in_c, :, :])
            erosion_result = torch.min(curr_result)
            return erosion_result

        def dilation_trimmed(curr_region, n, num_weight, in_c, r, c):
            #Element-wise addition between the current region and the filter.
            curr_result = curr_region * self.weight[num_weight, in_c, :, :]
            curr_result = torch.flatten(curr_result)
            curr_result = torch.sort(curr_result)
            curr_result = curr_result[0]
            num_removed = math.ceil(len(curr_result)*0.05)
            curr_result = curr_result[num_removed:]
            cur_result = curr_result[:len(curr_result)-num_removed]
            dilation_result = torch.max(curr_result)
            return dilation_result

        def erosion_trimmed(curr_region, n, num_weight, in_c, r, c):
            #Element-wise addition between the current region and the filter.
            curr_result = torch.abs(curr_region + self.weight[num_weight, in_c, :, :])
            curr_result = torch.flatten(curr_result)
            curr_result = torch.sort(curr_result)
            curr_result = curr_result[0]
            num_removed = math.ceil(len(curr_result)*0.05)
            curr_result = curr_result[num_removed:]
            cur_result = curr_result[:len(curr_result)-num_removed]
            erosion_result = torch.min(curr_result)
            return erosion_result

        def weighted_average(curr_region, n, num_weight, in_c, r, c):
            #Element-wise addition between the current region and the filter.
            curr_result = curr_region * self.weight[num_weight, in_c, :, :]
            sum_result = torch.sum(curr_result)
            sum_weights = torch.sum(self.weight[num_weight, in_c, :, :])
            weighted_result = sum_result/sum_weights
            return weighted_result

        #Looping through the image to apply the convolution operation.
        for n in range(0, x.shape[0]): # Loop through batch
            for in_c in range(0, self.size_in): # Loop through input channels
                num_weight = 0
                for out_c in range(0, self.size_out, len(self.operations)): # Loop through ouput channels
                    for r in range(0, x_padded.shape[2] - filter_size - 1 + 2*self.padding, self.stride): # Loop through height
                        for c in range(0, x_padded.shape[3] - filter_size - 1 + 2*self.padding, self.stride): # Loop through width
                            
                            """
                            Determining which operation corresponds to the filter iteration
                            represented as out channels.
                            """

                            if (1 in self.operations) or (3 in self.operations) or (4 in self.operations):
                                """
                                Getting the current region to get multiplied with the filter.
                                How to loop through the image and get the region based on 
                                the image and filer sizes is the most tricky part of convolution.
                                """
                                curr_region = x_padded[n, in_c, r:r+filter_size, c:c+filter_size]
                                dil_result = dilation_ordinary(curr_region, n, num_weight, in_c, r, c)

                                if 1 in self.operations:
                                    result[n, out_c, r, c] = dil_result
                                if (3 in self.operations) or (4 in self.operations) or (5 in self.operations):
                                    img_dilation[r, c] = dil_result
                                    #aux_dilation = img_dilation[r:r+filter_size, c:c+filter_size]

                            if (2 in self.operations) or (3 in self.operations) or (4 in self.operations):
                                """
                                Getting the current region to get multiplied with the filter.
                                How to loop through the image and get the region based on 
                                the image and filer sizes is the most tricky part of convolution.
                                """
                                curr_region = x_padded[n, in_c, r:r+filter_size, c:c+filter_size]
                                er_result = erosion_ordinary(curr_region, n, num_weight, in_c, r, c)
                              
                                if 2 in self.operations:
                                    if 1 in self.operations:
                                        result[n, out_c+1, r, c] = er_result
                                    else:
                                        result[n, out_c, r, c] = er_result
                                
                                if (3 in self.operations) or (4 in self.operations) or (5 in self.operations):
                                    img_erosion[r, c] = er_result
                                    #aux_erosion = img_erosion[r:r+filter_size, c:c+filter_size]
    
                    for r in range(0, x_padded.shape[2] - filter_size - 1 + 2*self.padding, self.stride): # Loop through height
                        for c in range(0, x_padded.shape[3] - filter_size - 1 + 2*self.padding, self.stride): # Loop through width       
                            aux_erosion = img_erosion[r:r+filter_size, c:c+filter_size]
                            if 3 in self.operations:
                                op_result = dilation_ordinary(aux_erosion, n, num_weight, in_c, r, c)

                                if (1 in self.operations) and (2 in self.operations):
                                    result[n, out_c+2, r, c] = op_result
                                elif (1 in self.operations) and not (2 in self.operations):
                                    result[n, out_c+1, r, c] = op_result
                                else:
                                    result[n, out_c, r, c] = op_result

                                if self.subtraction:
                                    if ("opening" in self.subtraction) and (5 in self.operations):
                                        img_opening[r, c] = op_result

                            aux_dilation = img_dilation[r:r+filter_size, c:c+filter_size]
                            if 4 in self.operations:
                                clos_result = erosion_ordinary(aux_dilation, n, num_weight, in_c, r, c)
                              
                                if (1 in self.operations) and (2 in self.operations) and (3 in self.operations):
                                    result[n, out_c+3, r, c] = clos_result
                                elif (1 in self.operations) and (2 in self.operations) and not(3 in self.operations):
                                    result[n, out_c+2, r, c] = clos_result
                                elif (1 in self.operations) and not (2 in self.operations) and (3 in self.operations):
                                    result[n, out_c+2, r, c] = clos_result
                                elif not (1 in self.operations) and (2 in self.operations) and (3 in self.operations):
                                    result[n, out_c+2, r, c] = clos_result
                                elif (1 in self.operations) and not (2 in self.operations) and not(3 in self.operations):
                                    result[n, out_c+1, r, c] = clos_result
                                elif not (1 in self.operations) and (2 in self.operations) and not(3 in self.operations):
                                    result[n, out_c+1, r, c] = clos_result
                                elif not (1 in self.operations) and not (2 in self.operations) and (3 in self.operations):
                                    result[n, out_c+1, r, c] = clos_result
                                else:
                                    result[n, out_c, r, c] = clos_result

                                if self.subtraction:
                                    if ("closing" in self.subtraction) and (5 in self.operations):
                                        img_closing[r, c] = clos_result
                    
                    if (5 in self.operations):
                        # Defining operation from which to subtract
                        if self.subtraction[0] == "dilation":
                            op1 = img_dilation[2:-2,2:-2]
                        elif self.subtraction[0] == "erosion":
                            op1 = img_erosion[2:-2,2:-2]
                        elif self.subtraction[0] == "opening":
                            op1 = img_opening
                        elif self.subtraction[0] == "closing":
                            op1 = img_closing
                        elif self.subtraction[0] == "original image":
                            op1 = x[n, in_c, :, :]

                        # Defining operation to be subtracted
                        if self.subtraction[1] == "dilation":
                            op2 = img_dilation[2:-2,2:-2]
                        elif self.subtraction[1] == "erosion":
                            op2 = img_erosion[2:-2,2:-2]
                        elif self.subtraction[1] == "opening":
                            op2 = img_opening
                        elif self.subtraction[1] == "closing":
                            op2 = img_closing
                        elif self.subtraction[1] == "original image":
                            op2 = x[n, in_c, :, :]
                    
                        # Perform subtraction
                        if len(self.operations) == 5:
                            result[n, out_c+4, :, :] = op1 - op2
                        elif len(self.operations) == 4:
                            result[n, out_c+3, :, :] = op1 - op2
                        elif len(self.operations) == 3:
                            result[n, out_c+2, :, :] = op1 - op2
                        elif len(self.operations) == 2:
                            result[n, out_c+1, :, :] = op1 - op2
                        else:
                            result[n, out_c, :, :] = op1 - op2

                    num_weight += 1
            result[n, :, :, :] = result[n, :, :, :] + torch.reshape(self.bias, (1,int((self.size_out/len(self.operations))*len(self.operations)),1,1))
        
        print("End of dilation layer")        
        return result