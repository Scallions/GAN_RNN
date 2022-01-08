import pytest
import data.tools as tools



class TestNC:
    """测试nc文件的加载
    划分：
        路径：
            路径不存在
            路径存在
        数据：
            数据有问题
            数据没问题
    """
    def test_ok_load(self):
        ds = tools.load_nc("./dataset/CSR_grid_DDK3.nc")
        assert ds.shape == (198, 181, 360)
        
    def test_wrong_path(self):
        with pytest.raises(FileNotFoundError):
            ds = tools.load_nc("./dataset/wrong.nc")