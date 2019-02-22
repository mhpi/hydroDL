import torch


class sigmaLoss(torch.nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean', prior=''):
        super(sigmaLoss, self).__init__()
        self.reduction = 'elementwise_mean'
        if prior == '':
            self.prior = None
        else:
            self.prior = prior.split('+')

    def forward(self, input, target):
        p = input[:, :, 0]
        s = input[:, :, 1]
        # s = input[-1, :, 1]
        t = target[:, :, 0]
        loc0 = t == p
        s[loc0] = 1
        # s.detach()
        if self.prior[0] == 'gauss':
            loss = torch.exp(-s).mul(torch.mul(p-t, p-t))/2+s/2
            lossMeanT = torch.mean(loss, dim=0)
        elif self.prior[0] == 'invGamma':
            c1 = float(self.prior[1])
            c2 = float(self.prior[2])
            nt = p.shape[0]
            loss = torch.exp(-s).mul(torch.mul(p-t, p-t)+c2/nt)/2+(1/2+c1/nt)*s
            loss[loc0] = 0
            lossMeanT = torch.mean(loss, dim=0)

        lossMeanB = torch.mean(lossMeanT)
        return lossMeanB
