import pytest
from data.smb import SmbTsDataset

class TestDataset:
    
    def test_ds(self):
        ds = SmbTsDataset("./dataset/CSR_grid_DDK3.nc")
        xs = next(iter(ds))
        assert xs[0].shape == (24, 181, 360)
        assert xs[1].shape == (12, 181, 360)