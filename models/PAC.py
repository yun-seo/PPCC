import torch

from .network import *
from utils.misc import fps
         
class PartAwareCompletion(nn.Module):
    def __init__(self, seg_num_all, enc, trm, dec, ref=None):
        super().__init__()
        
        self.enc_args = enc
        self.trm_args = trm
        self.dec_args = dec
        self.ref_args = ref
        
        self.enc = FeatureExtractor(**self.enc_args)
        self.trm = PartAwareTransformer(seg_num_all, **self.trm_args)
        self.dec = Decoder(**self.dec_args)
        self.ref = RefineModule(**self.ref_args)

    def only_gt_part(self, point, mask=None):
        new_point = point.clone().detach()
        if mask is not None:
            bs, seg_num_all = mask.shape
            if seg_num_all % 2 != 0:
                new_point = fps(new_point,int(new_point.shape[1]/seg_num_all)*seg_num_all)
            new_point = new_point.reshape(bs,seg_num_all,-1,3)
            with torch.no_grad():
                new_point = torch.cat([fps(new_point[bidx,mask[bidx]].reshape(1,-1,3), 2048) for bidx in range(bs)],0)
            return new_point
        else:
            return point

    def forward(self, pcd, mask, gt_mask=None):
        part_feats = self.enc(pcd)
        pts, feats = self.trm(part_feats, mask)
        part_pts = self.dec(pts, feats)
        new_part_pts = self.only_gt_part(part_pts, gt_mask)
        ref_pts0, ref_pts1 = self.ref(new_part_pts)
        return feats, (pts, part_pts), (ref_pts0, ref_pts1)
        
        