from model.gan.dcgan import DCGAN
from data.test import TestDataset
from data.smb import SmbDataset
import paddle
import paddle.io

## 准备训练数据


gan = DCGAN()

## 训练

# ds = TestDataset()
ds = SmbDataset("./dataset/CSR_grid_DDK3.nc")
dl = paddle.io.DataLoader(ds, batch_size=10, shuffle=True)

# for x in dl:
#     print(x.shape)
#     y = gan(x)
#     print(y.shape)
#     break

gan.train_gan(dl, 10)