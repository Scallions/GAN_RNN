import pytest
import paddle
from model.modules.map2hidden import map2hidden


@pytest.fixture
def smb_inp():
    # ds = SmbTsDataset("./dataset/CSR_grid_DDK3.nc")
    # xs = next(iter(ds))
    xs = paddle.rand([2, 24, 181, 360]) # b t i batch time dim
    return xs # shape 24, 181, 360

@pytest.fixture
def smb_hidden_inp():
    # ds = SmbTsDataset("./dataset/CSR_grid_DDK3.nc")
    # xs = next(iter(ds))
    xs = paddle.rand([2, 24, 180]) # b t i batch time dim
    return xs # shape 24, 181, 360

@pytest.fixture
def map2hidden_model():
    return map2hidden()

class TestModule:

    def test_map2hidden(self, smb_inp, map2hidden_model):
        assert smb_inp.shape == [2,24,181,360]
        out = map2hidden_model(smb_inp)
        assert out.shape == [2,24,180]

    def test_gen(self, smb_hidden_inp):
        from model.gan.generators.dcgenerator import DCGenerator
        out_c = 9
        gen = DCGenerator(smb_hidden_inp.shape[1],out_c)
        out = gen(smb_hidden_inp)
        assert out.shape == [2,out_c,181,360]


    def test_lstm(self, smb_inp):
        from model.rnn.lstm import LSTM
        lstm = LSTM(24)
        out = lstm(smb_inp)
        assert out.shape == [2, 24, 180]

    def test_dec(self):
        from model.gan.discriminators.dcdiscriminator import DCDiscriminator
        dc = DCDiscriminator(1)
        inp = paddle.rand([2,1,181,360])
        out = dc(inp)
        assert out.shape == [2,1]