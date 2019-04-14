import torch


class SigmaLoss(torch.nn.Module):
    def __init__(self, prior='gauss'):
        super(SigmaLoss, self).__init__()
        self.reduction = 'elementwise_mean'
        if prior == '':
            self.prior = None
        else:
            self.prior = prior.split('+')

    def forward(self, output, target):
        p0 = output[:, :, 0]
        s0 = output[:, :, 1]
        t0 = target[:, :, 0]
        mask = t0 == t0
        p = p0[mask]
        s = s0[mask]
        t = t0[mask]
        if self.prior[0] == 'gauss':
            loss = torch.exp(-s).mul((p - t)**2) / 2 + s / 2
        elif self.prior[0] == 'invGamma':
            c1 = float(self.prior[1])
            c2 = float(self.prior[2])
            nt = p.shape[0]
            loss = torch.exp(-s).mul(
                (p - t)**2 + c2 / nt) / 2 + (1 / 2 + c1 / nt) * s
        lossMean = torch.mean(loss)
        return lossMean


class RmseLoss(torch.nn.Module):
    def __init__(self):
        super(RmseLoss, self).__init__()

    def forward(self, output, target):
        p0 = output[:, :, 0]
        t0 = target[:, :, 0]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        loss = torch.sqrt(((p - t)**2).mean())
        return loss
