import paddle
from paddle import nn

class Conv2d(nn.Layer):
    """
    共用一个卷积核
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = nn.Conv2D(1,1, **kwargs)

    def forward(self, x: paddle.Tensor):
        b, t, w, h = x.shape
        x = x.reshape([b*t, 1, w, h])
        x = self.conv(x)
        w, h = x.shape[-2:]
        x = x.reshape([b, t, w, h])
        return x


if __name__ == "__main__":
    inp = paddle.rand([2, 24, 181, 360])
    conv = Conv2d(kernel_size=4, stride=1, padding=1)
    out = conv(inp)
    print(out.shape)