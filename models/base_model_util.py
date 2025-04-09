import torch
import torch.nn as nn
import torch.nn.functional as F

'''
basic shape:  [bs, feature_channel, point_num]
'''

class MlpConv(nn.Module):
    def __init__(self, input_channel, channels, activation_function=None):
        super(MlpConv, self).__init__()
        self.layer_num = len(channels)
        self.net = nn.Sequential()
        last_channel = input_channel
        for i, channel in enumerate(channels):   
            self.net.add_module('Conv1d_%d' % i, nn.Conv1d(last_channel, channel, kernel_size=1))
            if i != self.layer_num - 1:
                self.net.add_module('ReLU_%d' % i, nn.ReLU())
            last_channel = channel
        if activation_function != None:
            self.net.add_module('af', activation_function)

    def forward(self, x):
        return self.net(x)        

class PcnEncoder2(nn.Module):
    def __init__(self, input_channel=3, out_c=1024):
        super(PcnEncoder2, self).__init__()
        self.mlp_conv_1 = MlpConv(input_channel, [128, 256])
        self.mlp_conv_2 = MlpConv(512, [512, out_c])

    def forward(self, x):
        '''
        x : [B, N, 3]
        '''
        B, N, _ = x.shape
        x = x.permute(0, 2, 1)
        x = self.mlp_conv_1(x)

        x_max = torch.max(x, 2, keepdim=True).values
        x_max = x_max.repeat(1, 1, N) 
        x = torch.cat([x, x_max], 1)
        
        x = self.mlp_conv_2(x)
        
        x_max = torch.max(x, 2, keepdim=True).values
        return x_max
