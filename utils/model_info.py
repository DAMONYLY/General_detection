from copy import deepcopy

import torch
import torch.nn as nn
from thop import profile
from torch.nn import parameter

def get_model_info(model, tsize):

    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device, dtype=torch.double)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize * tsize / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}GFLOPs".format(params, flops)
    return info

def load_darknet_weights(self, weight_file, cutoff=52):
    
    "https://github.com/ultralytics/yolov3/blob/master/models.py"
    import numpy as np
    from model.layers.conv_module import Convolutional
    print("load darknet weights : ", weight_file)

    with open(weight_file, 'rb') as f:
        _ = np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)
    count = 0
    ptr = 0
    for m in self.modules():
        if isinstance(m, Convolutional):
            # only initing backbone conv's weights
            if count == cutoff:
                break
            count += 1

            conv_layer = m._Convolutional__conv
            if m.norm == "bn":
                # Load BN bias, weights, running mean and running variance
                bn_layer = m._Convolutional__norm
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b

                print("loading weight {}".format(bn_layer))
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

            print("loading weight {}".format(conv_layer))