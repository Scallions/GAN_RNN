import paddle
from paddle.fluid.layers.rnn import rnn
import paddle.nn as nn






class RNN_GAN:
    def __init__(self, inp_size: int, hidden_size: int) -> None:
        super().__init__()

        # rnn 编码
        self.rnn = nn.LSTM(inp_size, hidden_size, num_layers=2)
        self.gan = DCGAN(hidden_size, inp_size)

    def forward(self, x: paddle.Tensor) -> None:
        # 预测过程
        # rnn 预测
        x_h, _ = self.rnn(x)
        # gan 生成
        y_h = self.gan(x)
        return y_h

    def gan_criterion(pred, flag):
        pass


    def load_model(self, fp):
        """加载GAN网络

        Args:
            fp (str): GAN保存模型位置
        """
        gan_state = paddle.load(fp)
        self.gan.load_dict(gan_state)
        rnn_state = paddle.load(fp)
        self.rnn.load_dict(rnn_state)

    def train_rnn(self, ds: paddle.io.DataLoader):
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
        # 先训练gan


        # 再使用gan和rnn一起训练

        pass