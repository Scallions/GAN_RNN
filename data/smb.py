"""
表面物质平衡数据集
"""

from paddle.io import Dataset
from .tools import load_nc

"""
gan数据集
"""
class SmbDataset(Dataset):
    """
    gan 数据集
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.smb_data = load_nc(self.data_dir)
        self.smb_data = self.smb_data/self.smb_data.values.std()

    def __getitem__(self, idx):
        """按idx获取数据

        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        return self.smb_data[idx, :, :].to_numpy().reshape([1, 181, 360])

    def __len__(self):
        return self.smb_data.shape[0]


"""
时间序列数据集
"""
class SmbTsDataset(Dataset):
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
        return self.smb_data[idx:idx+24, :, :].to_numpy(), self.smb_data[idx+24:idx+36, :, :].to_numpy()

    def __len__(self):
        return self.smb_data.shape[0] - 35


if __name__ == '__main__':
    ds = SmbTsDataset("./dataset/CSR_grid_DDK3.nc")
    xs = next(iter(ds))
    print(xs[0].shape, xs[1].shape)