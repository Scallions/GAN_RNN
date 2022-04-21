

from ..libs import manager
import paddle
import paddle.nn as nn

from ..gan.discriminators.pwvdis import PWVDis

class P2PEncoder(nn.Layer):
    def __init__(self, p_in, t_in, c_t, hidden_size=512):
        super().__init__()
        self.pemb = nn.Sequential(
            nn.Conv2D(1, 1, 3, padding=1, stride=2),
            nn.BatchNorm(1),
            nn.LeakyReLU(),
            nn.Flatten(),
            # nn.Linear(p_in, hidden_size),
        )
        # self.pemb = PWVDis(p_in, hidden_size)
        self.temb = nn.Sequential(
            # nn.Linear(t_in, hidden_size),
            nn.Conv1D(c_t, hidden_size, 1),
        )
        self.lstm = nn.LSTM(hidden_size*2, hidden_size*2)

    def forward(self, p, t):
        B, T, W, H = p.shape
        p = p.reshape([B*T, 1, W, H])
        p = self.pemb(p)
        p = p.reshape([B, T, -1])
        t = t.transpose([0, 2, 1])
        t = self.temb(t)
        t = t.transpose([0, 2, 1])
        x = paddle.concat([p, t], axis=2)
        x, h = self.lstm(x)
        return h

from ..gan.generators.pwvgan import PWVGenerator

class P2PDecoder(nn.Layer):
    def __init__(self, hidden_size, c_t, out_w, out_h):
        super().__init__()
        self.temb = nn.Sequential(
            # nn.Linear(t_in, hidden_size),
            nn.Conv1D(c_t, hidden_size*2, 1),
        )
        self.lstm = nn.LSTM(hidden_size*2, hidden_size*2)
        # self.outmap = nn.Sequential(
        #     nn.Linear(hidden_size, out_w*out_h),
        #     nn.LeakyReLU(),
        # )
        self.outmap = PWVGenerator(hidden_size*2, 1, out_w=out_w, out_h=out_h)

    def forward(self, x, h):
        x = x.transpose([0, 2, 1])
        x = self.temb(x)
        x = x.transpose([0, 2, 1])
        x, _ = self.lstm(x, h)
        x = x.unsqueeze(3)
        B, T, C, _ = x.shape
        x = x.reshape([B*T, C, 1, 1])
        p = self.outmap(x)
        _, W, H = p.shape
        p = p.reshape([B, T, W, H])
        return p

@manager.MODELS.add_component
class P2P(nn.Layer):
    def __init__(self, p_in, t_in, out_w, out_h, c_t, hidden_size=512):
        super().__init__()
        self.encoder = P2PEncoder(p_in, t_in, c_t, hidden_size)
        self.decoder = P2PDecoder(hidden_size, c_t, out_w, out_h)
        # self.oulayer = nn.Sequential()

    def forward(self, p, t1, t2):
        h = self.encoder(p, t1)
        p2 = self.decoder(t2, h)
        # pout = self.outlayer(p, p2)
        return p2
