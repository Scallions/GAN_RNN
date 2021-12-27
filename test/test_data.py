import pytest
import data.ncload as ncload



class TestNC:
    """测试nc文件的加载
    """
    def test_load(self):
        ds = ncload.load_nc("./dataset/CSR_grid_DDK3.nc")
        assert ds.shape == (198, 181, 360)
        