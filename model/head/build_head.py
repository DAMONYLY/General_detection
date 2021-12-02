'''
2021年11月29日16:19:14
head 首先分共享不共享两种形式
共享，则直接建一个重复使用即可
不共享，则需要多建几次，再使用

其次，按构成，分为回归头，分类头，置信度头
无置信度头，则放在分类头一起
'''
import torch.nn as nn
from model.head.head import Head
from model.head.yolo_head import Yolo_head

def build_head(head, channel, num_anchor):
    if head == 'yolo_head':
        return Yolo_head()
    elif head == 'normal':
        reg_channel = 5
        cls_channel = 20
        return Head(channel_in=channel, channel_mid= channel, 
                    channel_out=[reg_channel, cls_channel], num_anchor=num_anchor)
    else:
        raise NotImplementedError
    
