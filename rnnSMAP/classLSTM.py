
import collections
import rnnSMAP
import torch


class optLSTM(collections.OrderedDict):
    def __init__(self, **kw):
        self['rootDB'] = rnnSMAP.kPath['DBSMAP_L3_Global']
        self['rootOut'] = rnnSMAP.kPath['OutSMAP_L3_Global']
        self['gpu'] = 0
        self['out'] = 'test'
        self['train'] = 'Globalv8f1'
        self['var'] = 'varLst_soilM'
        self['varC'] = 'varConstLst_Noah'
        self['target'] = 'SMAP_AM'
        self['syr'] = 2015
        self['eyr'] = 2016
        self['resume'] = 0
        self['hiddenSize'] = 256
        self['rho'] = 30
        self['rhoL'] = 30
        self['rhoP'] = 0
        self['dr'] = 0.5
        self['nbatch'] = 100
        self['nEpoch'] = 500
        self['saveEpoch'] = 100
        self['addFlag'] = 0
        if kw.keys() is not None:
            for key in kw:
                if key in self:
                    try:
                        self[key] = type(self[key])(kw[key])
                    except:
                        print('skiped '+key+': wrong type')
                else:
                    print('skiped '+key+': not in argument dict')


class LSTMModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, nLayer=1):
        super(LSTMModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.nLayer = nLayer
        self.lstm = torch.nn.LSTM(nx, hiddenSize, nLayer)
        self.dropout = torch.nn.Dropout(dr)
        self.linear = torch.nn.Linear(hiddenSize, ny)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hiddenSize),
                torch.zeros(1, 1, self.hiddenSize))

    def forward(self, x):
        if torch.cuda.is_available():
            h0 = torch.zeros(self.nLayer, x.size(
                0), self.hiddenSize, requires_grad=True).cuda()
        else:
            h0 = torch.zeros(self.nLayer, x.size(
                0), self.hiddenSize, requires_grad=True)

        # Initialize cell state
        if torch.cuda.is_available():
            c0 = torch.zeros(self.nLayer, x.size(
                0), self.hiddenSize, requires_grad=True).cuda()
        else:
            c0 = torch.zeros(self.nLayer, x.size(
                0), self.hiddenSize, requires_grad=True)
        out, (hn, cn) = self.lstm(x)
        out = self.dropout(out)
        out = self.linear(out)
        return out
