

from paddle.io import Dataset

from ...libs import manager


@manager.DATASETS.add_component
class PWVDataset(Dataset):
    pass