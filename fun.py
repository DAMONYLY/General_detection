import torch
import torch.nn as nn

a = torch.randn((4, 1))
print(a)

a = torch.sigmoid(a)
print(a)
target = torch.tensor([0.0, 1.0, 1.0, 0.0]).unsqueeze(1)
cir = nn.BCELoss(reduction='sum')
loss_all = cir(a, target)
loss_obj = cir(a*target, target)
loss_nobj = cir(a*(1- target), (1- target)*target)

print('loss_all', loss_all, loss_obj, loss_nobj)