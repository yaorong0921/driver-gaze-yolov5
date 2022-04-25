"""
The convolutional LSTM is adapted from
https://github.com/yaorong0921/Driver-Intention-Prediction/blob/master/models/convolution_lstm.py
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable



class Net(nn.Module):
    def __init__(self, gridwidth, gridheight):
        super().__init__()
        self.conv1 = nn.Conv2d(512, 16, (1, 1), stride=(1, 1))
        self.pool = nn.AdaptiveAvgPool2d((6,10))
        self.fc3 = nn.Linear(960, gridheight*gridwidth)

    def forward(self, x):
        x = torch.squeeze(x)
        x = x.float()
        x = self.conv1(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc3(x)
        return x


class LstmNet(nn.Module):
    def __init__(self, gridwidth, gridheight):
        super().__init__()
        self.conv1 = nn.Conv2d(512, 16, (1, 1), stride=(1, 1))

        self.pool = nn.AdaptiveAvgPool2d((6,10))

        self.lstm = nn.LSTM(
             input_size=16*6*10,
             hidden_size=256,
             num_layers=1,
             batch_first=True)

        self.fc3 = nn.Linear(256, gridheight*gridwidth)

    def forward(self, x):
        x = torch.squeeze(x)
        x = x.float()
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.pool(self.conv1(c_in))
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.lstm(r_in)
        return self.fc3(r_out[:, -1, :])



class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])))


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)

class ConvLSTMNet(nn.Module):
    def __init__(self, gridheight, gridwidth, seqlen):
        super().__init__()
        self.conv1 = nn.Conv2d(512, 128, (1, 1), stride=(1, 1))
        self.pool = nn.AdaptiveAvgPool2d((6,10))
        self.convlstm = ConvLSTM(input_channels=128, hidden_channels=[16], kernel_size=3, step=seqlen,
                        effective_step=[seqlen-1])

        self.fc3 = nn.Linear(960, gridheight*gridwidth)

    def forward(self, x):
        x = torch.squeeze(x)
        x = x.float()
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.conv1(c_in)
        c_out = self.pool(c_out)
        output_convlstm, _ = self.convlstm(c_out)
        x = output_convlstm[0]
        x = x.view(batch_size, timesteps, -1)
        x = self.fc3(x[:, -1, :])
        return x
