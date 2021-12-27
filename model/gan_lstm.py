from numpy import int_
import paddle
import paddle.nn as nn
from xarray.coding.cftimeindex import trailing_optional



class DCGAN(nn.Layer):
    def __init__(self, inp_size: int, out_size: int) -> None:
        super().__init__()
        self.G = nn.Conv2D()
        self.D = nn.Conv2D()
        self.epoch_gan = 10
        self.epoch_lstm = 10
        
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.G(x)
    
    

class LSTM_GAN(nn.Layer):
    def __init__(self, inp_size: int, hidden_size: int) -> None:
        super().__init__()
   
        # lstm 编码
        self.lstm = nn.LSTM(inp_size, hidden_size, num_layers=2)
        self.gan = DCGAN(hidden_size, inp_size)
    
    def forward(self, x: paddle.Tensor) -> None:
        
        x_h, _ = self.lstm(x)
        
        z = paddle.rand(x_h.shape)
        
        fake_y = 1
    
    def gan_criterion(pred, flag):
        pass
        
    
    def train_gan(self, ds):
        """训练GAN网络
        """
        opt_g = paddle.optimizer.Adam(0.1, parameters=self.gan.G.parameters())
        opt_d = paddle.optimizer.Adam(0.1, parameters=self.gan.D.parameters())
        for epoch in range(self.epoch_gan):
            for x in ds:
                # train D
                self.set_requires_grad(self.gan.G, False)
                self.set_requires_grad(self.gan.D, True)
                z = paddle.rand(shape=x.shape)
                f_x = self.gan.G(z)
                p_fx = self.gan.D(f_x)
                p_x = self.gan.D(x)
                ld_f = self.gan_criterion(p_fx, False)
                ld_r = self.gan_criterion(p_x, True)
                l = (ld_f + ld_r)*0.5
                l.backward()
                opt_d.step()
                opt_d.clear_grad()
                # train G
                self.set_requires_grad(self.gan.G, True)
                self.set_requires_grad(self.gan.D, False)
                p_fx = self.gan.D(f_x)
                l = self.gan_criterion(p_fx, True)
                l.backward()
                opt_g.step()
                opt_g.clear_grad()

    def set_requires_grad(self, nets, requires_grad: bool=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Args:
            nets (network list): a list of networks
            requires_grad (bool): whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.trainable = requires_grad             
                
    def load_gan(self, fp):
        """加载GAN网络

        Args:
            fp (str): GAN保存模型位置
        """
        gan_state = paddle.load(fp)
        self.gan.load_dict(gan_state)
        
    def train_lstm(self, ds: paddle.io.DataLoader):
        """训练整个网络

        Args:
            ds (dataloader)): 数据集
        """
        self.set_requires_grad(self.gan, False)
        epoch = 10
        opt = paddle.optimizer.Adam(0.1, parameters=self.parameters())
        loss = nn.MSELoss()
        for epoch in range(self.epoch_lstm):
            for x, y in ds:
                h_x, _ = self.lstm(x)
                p_x = self.gan.G(h_x)
                l = loss(p_x, y)
                l.backward()
                opt.step()
                opt.clear_grad()
                
    def train(self):
        self.train_gan()
        self.train_lstm()