import torch

a = torch.arange(60).reshape((3,4,5))
print(a)

b = torch.ones(3, 4)
c = (a[b, :])

print(c)
print(c.shape)