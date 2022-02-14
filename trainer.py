
from tqdm import tqdm
from paddle.io import DataLoader

from data.smb import SmbDataset, SmbTsDataset
from data.tools import split_ds
from model.gan_rnn import RNN_GAN

class Trainer:
    """模型训练
    """

    def __init__(self) -> None:

        ### 定义一些超参数
        self.data_dir = "./dataset/CSR_grid_DDK3.nc"
        self.model = RNN_GAN(10, 10) # 模型
        self.out_dir = None # 输出路径
        self.opt = None # 优化器
        self.lr = 0.001
        self.gan_epoch = 100
        self.rnn_epoch = 100
        self.ds = None # 总数据集
        self.train_ds = None # 训练集
        self.val_ds = None # 测试集
        self.loss_fn = None # 损失函数

        self.set_ds()

    def train(self):
        self.model.train(self.ds, self.train_ds, self.gan_epoch, self.rnn_epoch)

    def set_ds(self):
        """设置数据集
        """
        self.ds = SmbDataset(self.data_dir)
        ds = SmbTsDataset(self.data_dir)
        ts, vs = split_ds(ds)
        self.train_ds = DataLoader(ts, batch_size=8, shuffle=True, drop_last=True)
        self.val_ds = DataLoader(vs, batch_size=8)