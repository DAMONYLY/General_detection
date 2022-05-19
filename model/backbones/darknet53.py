import torch.nn as nn
from loguru import logger
from ..layers.conv_module import Convolutional
from ..layers.blocks_module import Residual_block


class Darknet53(nn.Module):

    def __init__(self):
        super(Darknet53, self).__init__()
        self.__conv = Convolutional(filters_in=3, filters_out=32, kernel_size=3, stride=1, pad=1, norm='bn',
                                    activate='leaky')

        self.__conv_5_0 = Convolutional(filters_in=32, filters_out=64, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_0 = Residual_block(filters_in=64, filters_out=64, filters_medium=32)

        self.__conv_5_1 = Convolutional(filters_in=64, filters_out=128, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_1_0 = Residual_block(filters_in=128, filters_out=128, filters_medium=64)
        self.__rb_5_1_1 = Residual_block(filters_in=128, filters_out=128, filters_medium=64)

        self.__conv_5_2 = Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_2_0 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_1 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_2 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_3 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_4 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_5 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_6 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_7 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)

        self.__conv_5_3 = Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_3_0 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_1 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_2 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_3 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_4 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_5 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_6 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_7 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)

        self.__conv_5_4 = Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_4_0 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)
        self.__rb_5_4_1 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)
        self.__rb_5_4_2 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)
        self.__rb_5_4_3 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)
        self.fpn_size = [256, 512, 1024]

    def forward(self, x):

        x = self.__conv(x)

        x = self.__conv_5_0(x)
        x = self.__rb_5_0(x)

        x = self.__conv_5_1(x)
        x = self.__rb_5_1_0(x)
        x = self.__rb_5_1_1(x)

        x = self.__conv_5_2(x)
        x = self.__rb_5_2_0(x)
        x = self.__rb_5_2_1(x)
        x = self.__rb_5_2_2(x)
        x = self.__rb_5_2_3(x)
        x = self.__rb_5_2_4(x)
        x = self.__rb_5_2_5(x)
        x = self.__rb_5_2_6(x)
        x = self.__rb_5_2_7(x)  # small, 8x

        xx = self.__conv_5_3(x)
        xx = self.__rb_5_3_0(xx)
        xx = self.__rb_5_3_1(xx)
        xx = self.__rb_5_3_2(xx)
        xx = self.__rb_5_3_3(xx)
        xx = self.__rb_5_3_4(xx)
        xx = self.__rb_5_3_5(xx)
        xx = self.__rb_5_3_6(xx)
        xx = self.__rb_5_3_7(xx)  # medium, 16x

        xxx = self.__conv_5_4(xx)
        xxx = self.__rb_5_4_0(xxx)
        xxx = self.__rb_5_4_1(xxx)
        xxx = self.__rb_5_4_2(xxx)
        xxx = self.__rb_5_4_3(xxx)  # large, 32x
        return x, xx, xxx  # [small, medium, large]

    def load_darknet_weights(self, weight_file, cutoff=52):
        "https://github.com/ultralytics/yolov3/blob/master/models.py"
        import torch
        import numpy as np

        logger.info("load darknet weights : ", weight_file)

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

                    logger.info("loading weight {}".format(bn_layer))
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

                logger.info("loading weight {}".format(conv_layer))
