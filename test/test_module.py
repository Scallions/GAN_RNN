import pytest
import paddle
from model.modules.map2hidden import map2hidden


@pytest.fixture
def smb_inp():
    # ds = SmbTsDataset("./dataset/CSR_grid_DDK3.nc")
    # xs = next(iter(ds))
    xs = paddle.rand([2, 1, 181, 360]) # b t i batch time dim
    return xs # shape 24, 181, 360


@pytest.fixture
def map2hidden_model():
    return map2hidden()

class TestModule:

    def test_map2hidden(self, smb_inp, map2hidden_model):
        assert smb_inp.shape == [2,1,181,360]
        out = map2hidden_model(smb_inp)
        assert out.shape == [2,1,180]
