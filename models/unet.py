import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
# from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

__all__ = ["UNet"]   # 在其他文件import *这个文件时，只有导入UNet模块

# from torch.nn.modules.module import T

class conv3x3_block_x2(nn.Module):
    '''
    特征提取，融合空间上下文信息
    感受野: 5x5(两个3x3叠加)
    保持尺寸不变（padding=1）
    '''
    def __init__(self, in_ch, out_ch):
        super(conv3x3_block_x2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class conv1x1(nn.Module):
    """
    1×1卷积层，只关注单个像素
    用于调整通道数，类似全连接但保持空间维度
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super(conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
    def forward(self, x):
        x = self.conv(x)
        return x

class upsample(nn.Module):
    """
    上采样模块（UNet解码器核心）
    功能: 深层特征上采样后与浅层特征拼接，再通过卷积融合
    """
    def __init__(self, in_ch, out_ch):
        super(upsample, self).__init__()
        self.conv1x1 = conv1x1(in_ch, out_ch)
        self.conv = conv3x3_block_x2(in_ch, out_ch)

    def forward(self, deep_x, low_x):
        '''
        :param deep_x: 深层特征（小尺度） ([1, 1024, 16, 16])
        :param low_x:  低层特征（较大尺寸） ([1, 512, 32, 32])
        :return:
        '''
        deep_x = F.interpolate(deep_x, scale_factor=2, mode='bilinear', align_corners=False)  # ([1, 1024, 32, 32])
        deep_x = self.conv1x1(deep_x)   # ([1, 512, 32, 32])
        x = torch.cat([deep_x, low_x], dim=1) # ([1, 1024, 32, 32])
        x = self.conv(x)      # 通过卷积融合深层、浅层特征  ([1, 512, 32, 32])
        return x

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.block1 = conv3x3_block_x2(3, 64)
        self.block2 = conv3x3_block_x2(64, 128)
        self.block3 = conv3x3_block_x2(128, 256)
        self.block4 = conv3x3_block_x2(256, 512)
        self.block_out = conv3x3_block_x2(512, 1024)
        self.upsample1 = upsample(1024, 512)
        self.upsample2 = upsample(512, 256)
        self.upsample3 = upsample(256, 128)
        self.upsample4 = upsample(128, 64)
        self.upsample_out = conv3x3_block_x2(64, num_classes)
        self._init_weight()

    def forward(self, x):
        # x (1, 3, 256, 256)
        block1_x = self.block1(x)   # (1, 64, 256, 256)
        x = self.maxpool(block1_x)  # (1, 64, 128, 128)
        block2_x = self.block2(x)   # (1, 128, 128, 128)
        x = self.maxpool(block2_x)  # (1, 128, 64, 64)
        block3_x = self.block3(x)   # (1, 256, 64, 64)
        x = self.maxpool(block3_x)  # (1, 256, 32, 32)
        block4_x = self.block4(x)   # (1, 512, 32, 32)
        x = self.maxpool(block4_x)  # (1, 512, 16, 16)
        x = self.block_out(x)       # (1, 1024, 16, 16)
        x = self.upsample1(x, block4_x)    # (1, 512, 32, 32)
        x = self.upsample2(x, block3_x)    # (1, 256, 64, 64)
        x = self.upsample3(x, block2_x)    # (1, 128, 128, 128)
        x = self.upsample4(x, block1_x)    # (1, 64, 256, 256)
        x = self.upsample_out(x)           # (1, num_classes, 256, 256)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == '__main__':
    from thop import profile
    device = torch.device('cpu')
    model = UNet(3)
    model = model.to(device)
    input = torch.randn(1, 3, 256, 256)
    input = input.to(device)
    flops, params = profile(model, inputs=(input, ))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')



