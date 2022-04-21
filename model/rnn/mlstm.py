

from turtle import forward
import paddle
import paddle.nn as nn

from ..libs import manager

class Encoder(nn.Layer):
    def __init__(self, c_in, hidden):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(c_in, hidden, 3)

    def forward(self, x):
        return self.lstm(x)

class Decoder(nn.Layer):
    def __init__(self, c_in, hidden, c_out):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(c_in, hidden, 2)
        self.linear = nn.Linear(hidden, c_out)

    def forward(self, x, h, c):
        x = self.lstm(x, h, c)
        x = self.linear(x)
        return x

@manager.MODELS.add_component
class Seq2seq(nn.Layer):
    def __init__(self, c_in, c_out, hidden):
        super(Seq2seq, self).__init__()
        self.encoder = Encoder(c_in, hidden)
        self.decoder = Decoder(c_in, hidden, c_out)

    def forward(self, x, y):
        x, h = self.encoder(x)
        x = self.decoder(y, h)
        return x

@manager.MODELS.add_component
class MLSTM(nn.Layer):

    def __init__(self, c_in, c_out, t_in, t_out, hidden_cells=2, hidden_size=512):
        """MLSTM

        Args:
            c_in (int): 输入特征数
            c_out (int): 输出特征数
            t_in (int): 输入时间长度
            t_out (int): 输出时间长度
        """
        super().__init__()
        self.lstm = nn.LSTM(c_in, hidden_size, hidden_cells)
        self.bn1 = nn.BatchNorm1D(t_in)
        self.relu1 = nn.LeakyReLU()
        self.out_conv = nn.Conv1D(t_in, t_out, 1)
        self.bn2 = nn.BatchNorm1D(t_out)
        self.relu2 = nn.LeakyReLU()
        self.out_layer = nn.Linear(hidden_size, c_out)

    def forward(self, x):
        x = self.lstm(x)[0]
        x = self.relu1(self.bn1(x))
        x = self.out_conv(x)
        x = self.relu2(self.bn2(x))

        x = self.out_layer(x)
        return x