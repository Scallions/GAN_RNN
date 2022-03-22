"""
假的数据集，测试用
"""

import paddle
from paddle.io import Dataset

"""
给gan使用的测试数据集
"""
class TestDataset(Dataset):

    def __init__(self):
        self.smb_data = paddle.rand([12, 1, 181, 360])

    def __getitem__(self, idx):
        """按idx获取数据

        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        return self.smb_data[idx, :, :]

    def __len__(self):
        return self.smb_data.shape[0]


"""
给lstm使用的测试数据集
"""
class TestTsDataset(Dataset):
    def __init__(self):
        self.smb_data = paddle.rand([180, 9, 20])

    def __getitem__(self, idx):
        """按idx获取数据

        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        return self.smb_data[idx:idx+24, :, :], self.smb_data[idx+24:idx+36, :, :]

    def __len__(self):
        return self.smb_data.shape[0] - 35

if __name__ == '__main__':
    ds = TestDataset()
    xs = next(iter(ds))
    print(xs[0].shape, xs[1].shape)