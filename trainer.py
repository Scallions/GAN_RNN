
from tqdm import tqdm
import paddle

class Trainer:
    """模型训练
    """
    
    def __init__(self) -> None:
        
        ### 定义一些超参数
        self.model = None # 模型
        self.out_dir = None # 输出路径
        self.opt = None # 优化器
        self.lr = 0.001
        self.epoch = 100
        self.train_ds = None # 训练集
        self.val_ds = None # 测试集
        self.loss_fn = None # 损失函数
        
    def train(self):
        
        for epoch in tqdm(range(self.epoch)):
            
            for x, y in tqdm(self.train_ds):
                
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
                
                # 更新梯度
                self.opt.clear_gradient()
                loss.back_ward()
                self.opt.step()
                
                # batch记录
                
            
            # epoch 记录
        
        
        # 训练结果记录