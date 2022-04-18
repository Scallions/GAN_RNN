import paddle.nn as nn
import paddle
from tqdm import tqdm

from .igan import IGAN

# from .generators.testgen import TGAN as GAN
# from .discriminators.testdis import TDIS as DIS
from .generators.dcgenerator import DCGenerator as GEN
from .discriminators.dcdiscriminator import DCDiscriminator as DIS
from ..libs import manager

@manager.MODELS.add_component
class DCGAN(IGAN):
    def __init__(self, inp_size: int=1, out_size: int=1) -> None:
        super().__init__()
        self.G = GEN(1,1)
        self.D = DIS(1)
        self.epoch_gan = 10
        self.epoch_lstm = 10
        self.loss = paddle.nn.BCELoss()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.G(x)

    def gan_criterion(self, p, l):
        # if f:
        #     # 对
        #     return (1-p).mean()
        # else:
        #     # 错
        #     return p.mean()
        return self.loss(p, l)

    def train_gan(self, ds, epoch_gan):
        """训练GAN网络
        """
        opt_g = paddle.optimizer.Adam(0.0002, parameters=self.G.parameters())
        opt_d = paddle.optimizer.Adam(0.0002, parameters=self.D.parameters())
        for epoch in tqdm(range(epoch_gan)):
            bar = tqdm(ds, leave=False)
            for bs, x in enumerate(bar):
                # train D
                self.set_requires_grad(self.G, False)
                self.set_requires_grad(self.D, True)
                b = x.shape[0]
                z = paddle.randn(shape=[b,1,9,20])
                f_x = self.G(z)
                p_fx = self.D(f_x)
                fl = paddle.full_like(p_fx, 0)
                p_x = self.D(x)
                l = paddle.full_like(p_x, 1)
                ld_f = self.gan_criterion(p_fx, fl)
                ld_r = self.gan_criterion(p_x, l)
                ld = (ld_f + ld_r) * 0.5
                ld.backward()
                opt_d.step()
                opt_d.clear_grad()
                # train G
                self.set_requires_grad(self.G, True)
                self.set_requires_grad(self.D, False)
                p_fx = self.D(f_x)
                lf = paddle.full_like(p_fx, 1)
                lg = self.gan_criterion(p_fx, lf)
                lg.backward()
                if bs % (max(len(ds)//10,1)) == 0:
                    bar.set_description(f"Ld: {ld.numpy().item():.3f}, Lg: {lg.numpy().item():.3f}")
                opt_g.step()
                opt_g.clear_grad()

        paddle.save(self.G.state_dict(), "checkpoints/G.pdmodel")