"""
将smb的map数据转换到隐空间
smb数据形状(181,360)
"""

import functools
import paddle
import paddle.nn as nn
from paddle.nn import BatchNorm2D

from .norm import build_norm_layer

def map2hidden():
    norm_type='instance'
    input_nc = 1
    norm_layer = build_norm_layer(norm_type)
    if type(norm_layer) == functools.partial:
        # no need to use bias as BatchNorm2d has affine parameters
        use_bias = norm_layer.func == nn.BatchNorm2D
    else:
        use_bias = norm_layer == nn.BatchNorm2D

    num_layers = 5

    sequence = [
        nn.Conv2D(input_nc,
                    1,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias_attr=use_bias),
        nn.LeakyReLU(0.2)
    ]


    # gradually increase the number of filters
    for n in range(1, num_layers):
        if norm_type == 'batch':
            sequence += [
                nn.Conv2D(1,
                            1,
                            kernel_size=4,
                            stride=2,
                            padding=1),
                BatchNorm2D(1),
                nn.LeakyReLU(0.2)
            ]
        else:
            sequence += [
                nn.Conv2D(1,
                            1,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias_attr=use_bias),
                norm_layer(1),
                nn.LeakyReLU(0.2)
            ]
    sequence += [
        nn.Flatten(start_axis=2),
        nn.Linear(11*22, 180)
    ]
    return nn.Sequential(*sequence)