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

@DATASETS.register_module()
class DA(data.Dataset):
    def __init__(self, config):
        self.subset = config.split
        self.category = config.category
        
        self.src = PartNet(config)
        self.tgt = PartUSSPA(config)
        
        if len(self.src) >= len(self.tgt):
            self.flag = True
        else:
            self.flag = False
        self.data_len = min(len(self.src), len(self.tgt))
        self.data_idx = self.data_len
        self.data_index = list(range(self.data_len))
        np.random.shuffle(self.data_index)

    def get_data_index(self):
        if self.data_idx >= self.data_len:
            self.data_idx = 0
            np.random.shuffle(self.data_index)
        res = self.data_index[self.data_idx]
        self.data_idx += 1
        return res

    def __len__(self):
        if self.subset != 'train':
            return len(self.tgt)
        else:
            return max(len(self.src), len(self.tgt))

    def __getitem__(self, index):
        if self.flag:
            src_data = self.src.__getitem__(index)
            tgt_data = self.tgt.__getitem__(self.get_data_index())
        else:
            src_data = self.src.__getitem__(self.get_data_index())
            tgt_data = self.tgt.__getitem__(index)
        return self.category, (src_data, tgt_data)
    
class PartNet(data.Dataset):
    def __init__(self, config):
        self.subset = config.split
        self.npoints = config.num_pts
        self.category = config.category
        self.data_path = config.data_path
        self.data_list = glob.glob(f'{self.data_path}/{self.category}/partnet/{self.subset}/*')
        print(f'[DATASET][PARTNET][{self.subset}] {len(self.data_list)} instances were loaded')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def clip_points(self, pc):
        idx = np.ones_like(pc[:, 0])
        for i in range(3):
            idx1 = pc[:, i] >= -1.0
            idx2 = pc[:, i] <= 1.0
            idx = np.logical_and(idx, idx1)
            idx = np.logical_and(idx, idx2)
        pc = pc[idx,:]
        return pc

    def __getitem__(self, idx):
        data = self.data_list[idx]
        data = IO.get(data).astype(np.float32)
        data[:,:3], _, _ = normalize(data[:,:3], get_arg=True)
        data = torch.from_numpy(data).float()
        return data
    
    def __len__(self):
        return len(self.data_list)
  
class PartUSSPA(data.Dataset):
    def __init__(self, config):
        self.subset = config.split
        self.npoints = config.num_pts
        self.category = config.category
        self.data_path = config.data_path
        self.file_list = sorted(glob.glob(f'{self.data_path}/{self.category}/usspa/{self.subset}/*'))
        print(f'[DATASET][USSPA][{self.subset}] {len(self.file_list)} instances were loaded')

    def clip_points(self, pc):
        idx = np.ones_like(pc[:, 0])
        for i in range(3):
            idx1 = pc[:, i] >= -1.0
            idx2 = pc[:, i] <= 1.0
            idx = np.logical_and(idx, idx1)
            idx = np.logical_and(idx, idx2)
        pc = pc[idx,:]
        return pc

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    def confidence_norm(self, pc):
        conf = pc[:,-1]
        cmax, cmin = conf.max(), conf.min()
        pc[:,-1] = (conf-cmin) / (cmax-cmin)
        return pc

    def __getitem__(self, idx):        
        sample = self.file_list[idx]
        data = IO.get(sample).astype(np.float32)
        gt = IO.get(sample.replace(f'{self.subset}',f'{self.subset}_gt')).astype(np.float32)
        
        name = sample.split('/')[-1].split('.')[0]
        gt[:,:3], center, max_scale = normalize(gt[:,:3], get_arg=True)
        data[:,:3] = normalize(data[:,:3], center=center, max_scale=max_scale)        
        
        data = self.clip_points(data)
        # some real point clouds are not match with gt after clip they contains 0 point
        # we repalce real point clouds with gt point clouds   
        if data.shape[0] <= 10:
            data = gt
        data = resample_pcd(data, self.npoints)
        data = self.confidence_norm(data)
        data = torch.from_numpy(data).float()
        gt = torch.from_numpy(gt).float()
        if self.subset != 'train':
            return (data, gt, name)
        else:
            return data

    def __len__(self):
        return len(self.file_list)
