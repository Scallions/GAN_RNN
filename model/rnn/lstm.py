import paddle
import functools
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F
from ..modules.norm import build_norm_layer
from ..modules.map2hidden import map2hidden

class LSTM(nn.Layer):
    def __init__(self):
        super().__init__()

        self.hidden = map2hidden()
        self.lstm = nn.LSTM(180, 180, 2)

    def forward(self, x):
        x = self.hidden(x)
        x, (_, _) = self.lstm(x)
        return x


if __name__ ==  '__main__':
    model = LSTM()
    inp = paddle.rand([2, 24, 180])
    print(model(inp).shape)
