import paddle
import paddle.nn as nn


class TGAN(nn.Layer):
    def __init__(self, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)


    def forward(self, x):
        res = paddle.rand([5,2,20,20])
        return res