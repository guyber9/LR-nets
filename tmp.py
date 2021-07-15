import torch
from torch.nn import functional as F

loaded = torch.load('my_tensors.pt')
x = loaded['x']
w = loaded['w']
# if torch.cuda.is_available():
#     torch.backends.cudnn.deterministic = True
z = F.conv2d(x, w, None, 1, 1, 1, 1)
v = torch.sqrt(z)

print("x is negative: " + str((x < 0).any()))
print("w is negative: " + str((w < 0).any()))
print("z (= Wx) is negative: " + str((z < 0).any()))
print("v is negative: " + str((v < 0).any()))
print("v isnan: " + str(torch.isnan(v).any()))
