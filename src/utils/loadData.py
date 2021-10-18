import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from src import config


class RhythmNetDataSet(Dataset):
    def __init__(self, data_roots):
        self.data_roots = data_roots

    def __len__(self):
        return len(self.data_roots)

    def __getitem__(self, index):
        st_map_root = config.PROJECT_ROOT + config.ST_MAP_ROOT
        target_hr_root = config.PROJECT_ROOT + config.HR_ROOT
        data_root = self.data_roots[index].replace('/', '_')
        st_map_file = st_map_root + data_root + ".npy"
        target_hr_file = target_hr_root + data_root + "hr.npy"
        st_maps = np.load(st_map_file)
        target_hr = np.load(target_hr_file)
        st_maps_shape = st_maps.shape
        st_maps = st_maps.reshape((-1, st_maps_shape[3], st_maps_shape[1], st_maps_shape[2]))

        return {
            "st_maps": torch.tensor(st_maps, dtype=torch.float),
            "target": torch.tensor(target_hr, dtype=torch.float)
        }
