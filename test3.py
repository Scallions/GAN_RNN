

from model.tft.tft import TFT
import paddle


model = TFT()

inp = paddle.rand([1,480,40])
out = model(inp)
print(out.shape)