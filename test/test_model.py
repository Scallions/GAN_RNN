import pytest
import paddle

from data.smb import SmbTsDataset
from model.rnn.lstm import LSTM
from model.gan.dcgan import DCGAN
from data.test import TestDataset


@pytest.fixture
def smb_hidden_inp():
    # ds = SmbTsDataset("./dataset/CSR_grid_DDK3.nc")
    # xs = next(iter(ds))
    xs = paddle.rand([2, 1, 180]) # b t i batch time dim
    return xs # shape 24, 181, 360

@pytest.fixture
def smb_inp():
    # ds = SmbTsDataset("./dataset/CSR_grid_DDK3.nc")
    # xs = next(iter(ds))
    xs = paddle.rand([2, 24, 181, 360]) # b t i batch time dim
    return xs # shape 24, 181, 360

@pytest.fixture
def lstm_model():
    return LSTM(24)

@pytest.fixture
def gan_model():
    from model.gan.dcgan import DCGAN
    return DCGAN(-1,-1)

class TestGAN:

    def test_output_shape(self, smb_hidden_inp, gan_model):
        # assert smb_inp.shape == (24, 181, 360)
        ## 检测inp shape
        assert smb_hidden_inp.shape == [2, 1, 180]
        out = gan_model(smb_hidden_inp)
        assert out.shape == [2,1, 181, 360]




class TestRNN:

    def test_lstm_out(self, smb_inp, lstm_model):
        out = lstm_model(smb_inp)
        assert out.shape == [2, 24, 180]


    def test_output_shape(self):
        pass