import torch
import torch.nn as nn
from timm.models.layers import DropPath,trunc_normal_
from .base_model_util import PcnEncoder2, MlpConv

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).expand(B, self.num_heads, N, N)
            attn = torch.where(mask == 0, attn.masked_fill(mask==0, -1e+9), attn*mask)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PartBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x, mask=None):
        if mask is not None:
            x_1 = self.attn(self.norm1(x), mask)
        else:
            x_1 = self.attn(self.norm1(x))
    
        x = x + self.drop_path(x_1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, num_feats):
        super().__init__()
        
        self.encoder = PcnEncoder2(3, num_feats)
        
    def forward(self, group_points):
        _, seg_num_all, _, _ = group_points.shape
        global_feats = torch.cat([self.encoder(group_points[:,i]) for i in range(seg_num_all)],-1)
        return global_feats.transpose(2,1)

class PartAwareTransformer(nn.Module):
    def __init__(self, seg_num_all, type, depth, num_head, num_query, mask_layer, trans_dim):
        super().__init__()
        
        self.type = type
        self.trans_dim = trans_dim
        self.mask_layer = mask_layer
        self.seg_num_all = seg_num_all
        
        self.encoder = nn.ModuleList([
            PartBlock(dim=trans_dim, num_heads=num_head) for i in range(depth[0])])

        self.increase_dim = nn.Sequential(
            nn.Conv1d(trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        
        self.num_query = num_query
        
        num_part = int(num_query / seg_num_all)
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * num_part)
        )
        
        self.mlp_query = nn.Sequential(
            nn.Conv1d(1024+3, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, trans_dim, 1),
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.part_token = nn.Parameter(torch.zeros(1, seg_num_all, trans_dim))
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.part_token, std=.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
           
    def forward(self, feats, mask):
        bs = feats.shape[0]
        pos = self.part_token.expand(bs,-1,-1) + self.cls_token.expand(bs,self.seg_num_all,-1)
        
        for i, blk in enumerate(self.encoder):
            if i < self.mask_layer:
                feats = blk(feats + pos, mask)
            else:
                feats = blk(feats + pos)
        
        global_feature = self.increase_dim(feats.transpose(1,2))
        
        coarse_point_cloud = self.coarse_pred(global_feature.reshape(bs*self.seg_num_all,-1)).reshape(bs, self.seg_num_all, -1, 3)
        num_part = coarse_point_cloud.shape[2]
        rebuild_query = torch.cat([torch.cat([coarse_point_cloud[:,i], global_feature[...,i].unsqueeze(1).expand(-1,num_part,-1)], -1) for i in range(self.seg_num_all)], 1)
        query = self.mlp_query(rebuild_query.transpose(2,1)).reshape(bs,-1,self.trans_dim)
        return coarse_point_cloud, query
    
class Decoder(nn.Module):
    def __init__(self, upn):
        super().__init__()

        self.mlp_1 = MlpConv(3, [256, 256, 384])
        self.mlp_2 = MlpConv(768+384, [512, 256, 384])
        self.UPN = upn
        self.mlp_3 = MlpConv(768+384, [512, 512, 3*self.UPN])

    def forward(self, pts, feats):

        bs, seg_num_all, num_part, _ = pts.shape
        pts = pts.reshape(bs, -1, 3)
        x = self.mlp_1(pts.permute(0, 2, 1)).transpose(2,1)
        x = x.reshape(bs, seg_num_all, num_part, -1)
        feats = feats.reshape(bs, seg_num_all, num_part, -1)
        x_max = torch.max(x, 2, keepdim=True).values
        x_list = [torch.cat([x[:,i], x_max[:,i].expand(-1,num_part,-1), feats[:,i]], -1) for i in range(seg_num_all)]
        
        x = [self.mlp_2(x_list[i].transpose(2,1)) for i in range(seg_num_all)]
        x_max = [torch.max(x[i], 2, keepdim=True).values for i in range(seg_num_all)]
        x_list = [torch.cat([x[i], x_max[i].expand(-1,-1,num_part), feats[:,i].transpose(2,1)], 1) for i in range(seg_num_all)]
       
        shift = [self.mlp_3(x_list[i]) for i in range(seg_num_all)]
        res = pts.unsqueeze(2).repeat([1, 1, self.UPN, 1])
        res = res.reshape(bs, -1, 3)
        shift = torch.cat([shift[i].reshape(bs,-1,3) for i in range(seg_num_all)], 1)
        res = res + shift
        return res

class RefineModule(nn.Module):
    def __init__(self, upn):
        super().__init__()
    
        self.lin_1 = MlpConv(3, [256, 256, 256])
        self.lin_2 = MlpConv(512, [256, 256, 256])
        self.lin_3 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512*3),
        )
        
        self.lin_4 = MlpConv(3+256, [256, 256, 256])
        self.UPN = upn
        self.lin_5 = MlpConv(256+256+3, [512, 512, 3*self.UPN])

    def forward(self, pts):
        
        bs, num, _ = pts.shape
        
        f0 = self.lin_1(pts.transpose(2,1))
        f0_max = torch.max(f0, 2, keepdim=True).values
        
        f1 = self.lin_2(torch.cat([f0, f0_max.repeat([1, 1, num])], 1))
        f1_max = torch.max(f1, 2, keepdim=True).values
        
        f2 = self.lin_3(f1_max.reshape(bs,-1))
        coarse = f2.reshape(bs,-1,3)
        n = coarse.shape[1]
        
        f3 = self.lin_4(torch.cat([coarse, f1_max.transpose(2,1).expand(-1,512,-1)],-1).transpose(2,1))
        f3_max = torch.max(f3, 2, keepdim=True).values
        f4 = torch.cat([coarse.permute(0, 2, 1), f3, f3_max.repeat([1, 1, n])], 1)
        shift = self.lin_5(f4)
        res = torch.unsqueeze(coarse, 2).repeat([1, 1, self.UPN, 1])
        res = torch.reshape(res, [bs, -1, 3])
        
        shift = shift.permute(0, 2, 1).reshape([bs, -1, 3])
        res = res + shift
        return coarse, res
