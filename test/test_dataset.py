import pytest
from data.smb import SmbDataset, SmbTsDataset

class TestDataset:
    """
    不是batch
    """
    def test_ds(self):
        ds = SmbTsDataset("./dataset/CSR_grid_DDK3.nc")
        xs = next(iter(ds))
        assert xs[0].shape == (24, 181, 360)
        assert xs[1].shape == (12, 181, 360)

    def test_gands(self):
        ds = SmbDataset("./dataset/CSR_grid_DDK3.nc")
        xs = next(iter(ds))
        assert xs.shape == (1, 181, 360)