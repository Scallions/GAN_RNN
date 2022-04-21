
from model.p2p.p2p import P2P
import paddle

model = P2P(p_in=24*7, t_in=24*7, out_w=64, out_h=31, c_t=35, hidden_size=512)


inp = paddle.rand([2, 24*7, 31, 64])
inpt = paddle.rand([2, 24*7, 35])
outt = paddle.rand([2, 7, 35])

out = model(inp, inpt, outt)
print(out.shape)
paddle.summary(model, [(1,24*7,31,64),(1,24*7,35),(1,7,35)])