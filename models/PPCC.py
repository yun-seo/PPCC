import torch
from torch import nn

from extensions.chamfer_dist import ChamferDistanceL1, PatialChamferDistanceL1

from .build import MODELS
from tools.builder import *
from .base_model_util import *

from utils.loss import part_loss
from utils.loss_util import BasicLoss

from .PAC import PartAwareCompletion
from .PBD import PartBasedDecoomposition

class PPCCLoss(BasicLoss):
    def __init__(self, config):
        super().__init__()
        
        self.category = config.category
        self.seg_num_all = config.seg_num_all
        
        self.lw = config.loss.weight
        self.loss_name = config.loss.name
        self.loss_num = len(self.loss_name)
        
        self.part = part_loss()
        self.cd1 = ChamferDistanceL1()
        self.ucd1 = PatialChamferDistanceL1()
    
    def batch_forward(self, outputs, data):
        
        __E = 1e-8
        _, points, disc_feats = outputs
        gt, tgt_partial = data
        gt, tgt_partial = gt.cuda(), tgt_partial.cuda()
        
        _, _, _, g_fake, d_fake, d_real, \
        src_dec0, src_dec1, src_ref0, src_ref1, tgt_dec1 = self.lw
        
        # discriminator loss
        df_src, df_tgt = disc_feats
        g_fake_loss, d_fake_loss, d_real_loss = 0., 0., 0.
        for i in range(self.seg_num_all):
            g_fake_loss += -torch.log(df_tgt[:,i]+__E)
            d_fake_loss += -torch.log(1-df_tgt[:,i]+__E)
            d_real_loss += -torch.log(df_src[:,i]+__E)
        
        # prediction
        src_pts, ref_src_pts, tgt_pts, _ = points
        src_pts0, src_pts1 = src_pts
        _, tgt_pts1 = tgt_pts

        # synthetic, decoder output
        bs = src_pts0.shape[0]
        src_loss0 = self.part(src_pts0.reshape(bs,-1,3), gt, part='part_coarse', seg_num_all=self.seg_num_all)
        src_loss1 = self.part(src_pts1, gt, part='part_dense', seg_num_all=self.seg_num_all)
        
        # real
        tgt_loss1 = self.ucd1(tgt_pts1, tgt_partial[...,:3].contiguous())
        
        # refinement module
        ref_src_pts0, ref_src_pts1 = ref_src_pts
        ref_src_loss0 = self.cd1(ref_src_pts0, gt[...,:3].contiguous())
        ref_src_loss1 = self.cd1(ref_src_pts1, gt[...,:3].contiguous())
        
        loss_g = g_fake*g_fake_loss + src_dec0*src_loss0 + src_dec1*src_loss1 + tgt_dec1*tgt_loss1
        
        loss_d = d_fake*d_fake_loss + d_real*d_real_loss
        
        loss_r = src_ref0*ref_src_loss0 + src_ref1*ref_src_loss1
        
        return [loss_g, loss_d, loss_r, g_fake_loss, d_fake_loss, d_real_loss, 
                src_loss0, src_loss1, ref_src_loss0, ref_src_loss1, tgt_loss1]

class PPCCDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.d_f = MlpConv(384, [64, 64, 1])
        self.seg_num_all = config.seg_num_all

    def discriminate_feature(self, f):
        d_f = self.d_f(f)
        d_f = torch.sigmoid(d_f)
        d_f = d_f[:,0,0]
        return d_f

    def forward(self, feats):
        
        src_feats, tgt_feats = feats
        bs, _, c = src_feats.shape
        
        src_feats = src_feats.reshape(bs,self.seg_num_all,-1,c)
        tgt_feats = tgt_feats.reshape(bs,self.seg_num_all,-1,c)
        feats = torch.cat([src_feats, tgt_feats], 0)
        df = torch.cat([self.discriminate_feature(feats[:,i].max(1)[0].unsqueeze(-1)).unsqueeze(-1) for i in range(self.seg_num_all)], -1)
        df_src, df_tgt = torch.split(df, [bs, bs], 0)
        return df_src, df_tgt

class PPCCGenerator(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        
        self.PBD_args = config.PBD
        self.PAC_args = config.PAC
        
        self.category = config.category
        self.seg_num_all = config.seg_num_all
        
        self.PBD = PartBasedDecoomposition(self.seg_num_all, self.PAC_args.trm.type, **self.PBD_args)
        self.PAC = PartAwareCompletion(self.seg_num_all, **self.PAC_args)

    def forward(self, data, validation=False):
        src, tgt = data

        if self.training:
            src, gt_mask = src
            
            part_src_set, src_mask = self.PBD(src)
            part_tgt_set, tgt_mask = self.PBD(tgt)

            src_feats, src_pts, ref_src_pts = self.PAC(part_src_set, src_mask, gt_mask)
            tgt_feats, tgt_pts, ref_tgt_pts = self.PAC(part_tgt_set, tgt_mask)
            return (src_feats, tgt_feats), (src_pts, ref_src_pts, tgt_pts, ref_tgt_pts)
        
        elif validation:
            part_src_set, src_mask = self.PBD(src)
            part_tgt_set, tgt_mask = self.PBD(tgt)
            
            _, _, ref_src_pts = self.PAC(part_src_set, src_mask)
            _, _, ref_tgt_pts = self.PAC(part_tgt_set, tgt_mask)
            return (ref_src_pts, ref_tgt_pts)
        
        else:
            part_set, mask = self.PBD(tgt)
            _, _, ref_pts = self.PAC(part_set, mask)
            return ref_pts

@MODELS.register_module()
class PPCC(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        
        self.generator = PPCCGenerator(config)
        self.discriminator = PPCCDiscriminator(config)
        self.loss = PPCCLoss(config)

    def forward(self, data, validation=False):
        
        if self.training:
            feats, points = self.generator(data)
            disc_feats = self.discriminator(feats)
            return feats, points, disc_feats
        else:
            points = self.generator(data, validation=validation)
            return points
            