""" 
表面物质平衡数据集
"""

from paddle.io import Dataset
from data.tools import load_nc

class SmbDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.smb_data = load_nc(self.data_dir)
        
    def __getitem__(self, idx):
        """按idx获取数据

        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        return self.smb_data[idx:idx+12, :, :], self.smb_data[idx+12, :, :]
    
    def __len__(self):
        return self.smb_data.shape[0] - 12