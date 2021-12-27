import pytest
import xarray as xr



class TestNC:
    """测试nc文件的加载
    """
    def test_load(self):
        ds = xr.load_dataarray("./dataset/CSR_grid_DDK3.nc")
        print(ds.shape)
        