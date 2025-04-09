import glob
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS

def normalize(pcd, get_arg=False, center=None, max_scale=None):
    if center is None or max_scale is None:
        maxs = np.max(pcd, 0, keepdims=True)
        mins = np.min(pcd, 0, keepdims=True)
        center = (maxs+mins)/2
        scale = (maxs-mins)/2
        max_scale = np.max(scale)
    pcd = pcd - center
    pcd = pcd / max_scale
    if get_arg:
        return pcd, center, max_scale
    else:  
        return pcd

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd[:,:3].shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]

def clip_points(pc):
    ori_pc = pc
    pc = pc[:,:3]
    idx = np.ones_like(pc[:, 0])
    for i in range(3):
        idx1 = pc[:, i] >= -1.0
        idx2 = pc[:, i] <= 1.0
        idx = np.logical_and(idx, idx1)
        idx = np.logical_and(idx, idx2)
    pc = ori_pc[idx,:]
    return pc

@DATASETS.register_module()
class PartUSSPA(data.Dataset):
    def __init__(self, config):
        
        self.subset = config.split
        self.npoints = config.num_pts
        self.category = config.category
        self.data_path = config.data_path
        self.file_list = sorted(glob.glob(f'{self.data_path}/{self.category}/usspa/{self.subset}/*'))
        print(f'[DATASET][USSPA][{self.subset}] {len(self.file_list)} instances were loaded')

    def confidence_norm(self, pc):
        conf = pc[:,-1]
        cmax, cmin = conf.max(), conf.min()
        pc[:,-1] = (conf-cmin) / (cmax-cmin)
        return pc

    def __getitem__(self, idx):
        
        sample = self.file_list[idx]
        name = sample.split('/')[-1].split('.')[0]
        
        data = IO.get(sample).astype(np.float32)
        gt = IO.get(sample.replace(f'{self.subset}',f'{self.subset}_gt')).astype(np.float32)
        
        gt[:,:3], center, max_scale = normalize(gt[:,:3], get_arg=True)
        data[:,:3] = normalize(data[:,:3], center=center, max_scale=max_scale)        
        
        data = clip_points(data)
        if data.shape[0] <= 10:
            data = gt

        data = resample_pcd(data, self.npoints)
        data = self.confidence_norm(data)
        data = torch.from_numpy(data).float()
        gt = torch.from_numpy(gt).float()
        if self.subset != 'train':
            return self.category, (data, gt, name)
        else:
            return data

    def __len__(self):
        return len(self.file_list)

