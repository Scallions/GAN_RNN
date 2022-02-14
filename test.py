import paddle

from data.smb import SmbTsDataset
from model.rnn.lstm import LSTM
from model.gan.dcgan import DCGAN
from data.test import TestDataset

gan = DCGAN(1,1)
ds = TestDataset()
from paddle.io import DataLoader
ds = DataLoader(ds)
# gan.train_gan(ds, 1)

# (24, 181, 360)

## test discriminator
from model.gan.discriminators.dcdiscriminator import DCDiscriminator
dc_disc = DCDiscriminator(1)
inp = paddle.rand([2,1,181,360])
out = dc_disc(inp)
print(out.shape)

## test generator
from model.gan.generators.dcgenerator import DCGenerator
dc_gen = DCGenerator(1, 1)
inp = paddle.rand([2,1,9,20])
out = dc_gen(inp)
print(out.shape)

## test dcgan
from model.gan.dcgan import DCGAN
dcgan = DCGAN(-1,-1)
inp = paddle.rand([2,1,180])
out = dcgan.forward(inp)
print(out.shape)
print(dcgan.gan_criterion(dcgan.D(out), False))

## test lstm
from model.rnn.lstm import LSTM
lstm = LSTM()
inp = paddle.rand([2,1,181,360])
out = lstm(inp)
print(out.shape)
