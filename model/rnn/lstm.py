import paddle
import functools
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import BatchNorm2D
from ..modules.norm import build_norm_layer

class LSTM(nn.Layer):
    def __init__(self):
        super().__init__()
        norm_type='instance'
        input_nc = 24
        norm_layer = build_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:
            # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.BatchNorm2D
        else:
            use_bias = norm_layer == nn.BatchNorm2D

        kw = 4
        padw = 1

        sequence = [
            nn.Conv2D(input_nc,
                      1,
                      kernel_size=kw,
                      stride=2,
                      padding=padw,
                      bias_attr=use_bias),
            nn.LeakyReLU(0.2)
        ]

        n_downsampling = 3

        # gradually increase the number of filters
        for n in range(1, n_downsampling):
            if norm_type == 'batch':
                sequence += [
                    nn.Conv2D(1,
                              1,
                              kernel_size=kw,
                              stride=2,
                              padding=padw),
                    BatchNorm2D(1),
                    nn.LeakyReLU(0.2)
                ]
            else:
                sequence += [
                    nn.Conv2D(1,
                              1,
                              kernel_size=kw,
                              stride=2,
                              padding=padw,
                              bias_attr=use_bias),
                    norm_layer(1),
                    nn.LeakyReLU(0.2)
                ]

        self.conv = nn.Sequential(*sequence)
        self.lstm = nn.LSTM(180, 180, 2)

    def forward(self, x):
        x = self.conv
        x, (_, _) = self.lstm(x)
        return x


if __name__ ==  '__main__':
    model = LSTM()
    inp = paddle.rand([2, 24, 180])
    print(model(inp).shape)
