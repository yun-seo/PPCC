import torch
from torch import nn
from extensions.chamfer_dist import ChamferDistanceL1

class part_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.loss_func = ChamferDistanceL1()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, ret, gt, part=None, seg_num_all=None):
        
        bs, _, _ = gt.shape  
        if part == 'part_coarse':
            coarse = ret.reshape(bs,seg_num_all,-1,3)
            part_loss = 0.
            for i in range(bs):
                uni_part = torch.unique(gt[i][:,3])
                loss = 0.
                for p in uni_part:
                    part_pred = coarse[i][p.int().item()].unsqueeze(0)
                    part_gt = gt[i][gt[i][:,3]==p.int().item()].unsqueeze(0)
                    loss += self.loss_func(part_pred, part_gt[...,:3].contiguous())
                part_loss += loss / len(uni_part)
            return part_loss / bs
        elif part == 'part_dense':
            fine = ret.reshape(bs,seg_num_all,-1,3)
            part_loss = 0.
            for i in range(bs):
                uni_part = torch.unique(gt[i][:,3])
                loss = 0.
                for p in uni_part:
                    part_pred = fine[i][p.int().item()].unsqueeze(0)
                    part_gt = gt[i][gt[i][:,3]==p.int().item()].unsqueeze(0)
                    loss += self.loss_func(part_pred, part_gt[...,:3].contiguous())
                part_loss += loss / len(uni_part)
            return part_loss / bs
        elif part == 'eval_part_coarse':
            coarse = ret[0].reshape(bs,seg_num_all,-1,3)
            for i in range(bs):
                uni_part = torch.unique(gt[i][:,3])
                loss = torch.zeros((4)).cuda()
                for p in range(seg_num_all):
                    if p in uni_part:
                        part_pred = coarse[i][p].unsqueeze(0)
                        part_gt = gt[i][gt[i][:,3]==p].unsqueeze(0)
                        loss[p] = self.loss_func(part_pred, part_gt[...,:3].contiguous())
                    else:
                        loss[p] = 0.
            return loss.unsqueeze(0)
        elif part == 'eval_part_dense':
            fine = ret[1].reshape(bs,seg_num_all,-1,3)
            for i in range(bs):
                uni_part = torch.unique(gt[i][:,3])
                loss = torch.zeros((4)).cuda()
                for p in range(seg_num_all):
                    if p in uni_part:
                        part_pred = fine[i][p].unsqueeze(0)
                        part_gt = gt[i][gt[i][:,3]==p].unsqueeze(0)
                        loss[p] = self.loss_func(part_pred, part_gt[...,:3].contiguous())
                    else:
                        loss[p] = 0.
            return loss.unsqueeze(0)
        else:
            raise NotImplementedError('loss is not implemented')
