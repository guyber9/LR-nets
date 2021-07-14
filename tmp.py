import torch
from torch.nn import functional as F

# loaded = torch.load('my_tensors.pt')
# print(loaded)
# x = loaded['x']
# w = loaded['w']
loaded1 = torch.load('my_tensors1.pt')
loaded2 = torch.load('my_tensors2.pt')
x = loaded1['x']
w = loaded2['w']

z = F.conv2d(x, w, None, 1, 1, 1, 1)
v = torch.sqrt(z)
print("v: " + str(v))
print("z: " + str(z))

print("x: " + str(x))
print("x < 0: " + str(x < 0))

print("w: " + str(w))
print("w < 0: " + str(w < 0))

print("x is positive: " + str((x < 0).any()))
print("w is positive: " + str((w < 0).any()))
print("z is positive: " + str((z < 0).any()))
print("v is positive: " + str((v < 0).any()))
print("v isnan: " + str(torch.isnan(v).any()))
