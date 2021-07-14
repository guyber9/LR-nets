import torch
from torch.nn import functional as F

loaded = torch.load('my_tensors.pt')
x = loaded['x']
w = loaded['w']

z1 = F.conv2d(x, w, None, 1, 1, 1, 1)
v = torch.sqrt(z1)
print("v isnan: " + str(torch.isnan(v).any()))
print("x is positive: " + str((x > 0).any()))
print("x > 0: " + str(x > 0))
