import xarray as xr



def load_nc(fp: str) -> xr.DataArray:
    """加载nc文件以后可能会最一些预处理

    Args:
        fp (str): 文件的路径

    Returns:
        xr.DataArray: 文件保存的数据，一般是（t，lat，lon）
    """
    ds = xr.load_dataarray(fp)
    return ds


def split_ds(ds):
    """将数据集分割为训练集和测试集

    Args:
        ds ([type]): [description]

    Returns:
        [type]: [description]
    """
    ts = None
    vs = None
    return ts, vs