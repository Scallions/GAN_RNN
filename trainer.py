
from tqdm import tqdm
from paddle.io import DataLoader

from data.smb import SmbDataset
from data.tools import split_ds

class Trainer:
    """模型训练
    """
    
    def __init__(self) -> None:
        
        ### 定义一些超参数
        self.data_dir = "./dataset/CSR_grid_DDK3.nc"
        self.model = None # 模型
        self.out_dir = None # 输出路径
        self.opt = None # 优化器
        self.lr = 0.001
        self.epoch = 100
        self.train_ds = None # 训练集
        self.val_ds = None # 测试集
        self.loss_fn = None # 损失函数
        
        self.set_ds()
        
    def train(self):
        
        for epoch in tqdm(range(self.epoch)):
            
            for batch, (x, y) in tqdm(enumerate(self.train_ds)):
                
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
                
                # 更新梯度
                self.opt.clear_gradient()
                loss.back_ward()
                self.opt.step()
                
                # batch记录
                
            
            # epoch 记录
        
        
        # 训练结果记录
        
    def set_ds(self):
        """设置数据集
        """
        ds = SmbDataset(self.data_dir)
        ts, vs = split_ds(ds)
        self.train_ds = DataLoader(ts, batch_size=8, shuffle=True, drop_last=True)
        self.val_ds = DataLoader(vs, batch_size=8)