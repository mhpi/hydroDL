import torch
import rnnSMAP

# create a test dataset
nt = 3
ngrid = 2
nx = 2
ny = 1
nh = 4

# LSTM
# x = torch.rand(nt, ngrid, nx).cuda()
# y = torch.rand(nt, ngrid, ny).cuda()

# n1 = torch.nn.LSTM(nx, 4, 1).cuda()
# n2 = torch.nn.Dropout(0.5).cuda()

# o1, (hn, cn) = n1(x)
# o2 = n2(o1)

# Linear
x = torch.rand(ngrid, nx)
y = torch.rand(ngrid, ny)
n1=torch.nn.Linear(nx,ny)
n2=torch.nn.Dropout(0.5)
o1=n1(x)
o2=n2(o1)

