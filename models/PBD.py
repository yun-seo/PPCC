import torch
from torch import nn
import utils.misc as misc

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    try:
        B, N, _ = src.shape
    except:
        import pdb; pdb.set_trace()
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist  

class PartBasedDecoomposition(nn.Module):
    def __init__(self, seg_num_all, type, num_fps, conf_thres):
        super().__init__()
    
        self.type = type
        self.num_fps = num_fps
        self.conf_thres = conf_thres
        self.seg_num_all = seg_num_all

    def forward(self, x):
        
        bs = x.shape[0]
        points = x[...,:3]
        label = x[...,3]
        group_points = torch.zeros((bs, self.seg_num_all, self.num_fps, 3)).to(points.device)
        
        # synthetic
        if x.size(2) == 4:
            masks  = torch.cat([((label == i).sum(1) >= self.num_fps).unsqueeze(-1) for i in range(self.seg_num_all)], 1)
            for bidx, m in enumerate(masks):
                for sidx, s in enumerate(m):
                    if s.item():
                        fps_point = misc.fps(points[bidx,(label[bidx]==sidx)].unsqueeze(0), self.num_fps).squeeze(0)
                        group_points[bidx,sidx] = fps_point
                    else: continue
        
        # real
        elif x.size(2) == 5:
            masks  = torch.cat([((label == i).sum(1) >= self.num_fps).unsqueeze(-1) for i in range(self.seg_num_all)], 1)
            conf_masks = x[...,4] > self.conf_thres
            for bidx, m in enumerate(masks):
                for sidx, s in enumerate(m):
                    if s.item():
                        if points[bidx,torch.logical_and(conf_masks[bidx], label[bidx]==sidx)].size(0) >= self.num_fps:
                            fps_point = misc.fps(points[bidx,torch.logical_and(conf_masks[bidx], label[bidx]==sidx)].unsqueeze(0), self.num_fps).squeeze(0)
                            group_points[bidx,sidx] = fps_point
                        else: continue
                    else: continue
        return group_points, masks