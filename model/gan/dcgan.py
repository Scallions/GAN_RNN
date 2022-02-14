import paddle.nn as nn
import paddle


from .igan import IGAN

# from .generators.testgen import TGAN as GAN
# from .discriminators.testdis import TDIS as DIS
from .generators.dcgenerator import DCGenerator as GAN
from .discriminators.dcdiscriminator import DCDiscriminator as DIS

class DCGAN(IGAN):
    def __init__(self, inp_size: int, out_size: int) -> None:
        super().__init__()
        self.G = GAN(1,1)
        self.D = DIS(1)
        self.epoch_gan = 10
        self.epoch_lstm = 10

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.G(x)

    def gan_criterion(self, p, f):
        if f:
            # 对
            return (1-p).mean()
        else:
            # 错
            return p.mean()

    def train_gan(self, ds, epoch_gan):
        """训练GAN网络
        """
        opt_g = paddle.optimizer.Adam(0.1, parameters=self.G.parameters())
        opt_d = paddle.optimizer.Adam(0.1, parameters=self.D.parameters())
        for epoch in range(epoch_gan):
            for x in ds:
                # train D
                self.set_requires_grad(self.G, False)
                self.set_requires_grad(self.D, True)
                z = paddle.rand(shape=x.shape)
                f_x = self.G(z)
                p_fx = self.D(f_x)
                p_x = self.D(x)
                ld_f = self.gan_criterion(p_fx, False)
                ld_r = self.gan_criterion(p_x, True)
                l = (ld_f + ld_r)*0.5
                l.backward()
                opt_d.step()
                opt_d.clear_grad()
                # train G
                self.set_requires_grad(self.G, True)
                self.set_requires_grad(self.D, False)
                p_fx = self.D(f_x)
                l = self.gan_criterion(p_fx, True)
                l.backward()
                opt_g.step()
                opt_g.clear_grad()

