import torch
import torch.nn as nn
import torch.nn.functional as F
from vgg import vgg16

class _FCNHead(nn.Module):
    '''
    FCN分割头模块 - 将主干网络提取的特征映射到分割类别空间
    作用:
        1. 将高维特征(512维)降维到与类别数匹配的维度
        2. 通过额外的卷积层进一步融合特征，提升分割精度
        3. 作为FCN模型中的解码器头部
    参数:
        in_channels: 输入特征图的通道数，通常为512(VGG最后一层输出)
        out_channels: 输出通道数，等于分割的类别数(如PASCAL VOC的21类)
        norm_layer: 归一化层类型，默认为BatchNorm2d
    '''
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()

        # 中间通道数：将输入通道数压缩到1/4，作为中间表示
        inter_channels = in_channels // 4

        # 定义分割头的网络块
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=inter_channels,
                      kernel_size=3,  # 使用3×3卷积核(非1×1)可以融合空间上下文信息
                      padding=1,      # padding=1保持特征图尺寸不变
                      bias=False),    # bias=False因为后续有BatchNorm，bias参数冗余

            norm_layer(inter_channels),   # 2. 归一化层：加速训练，稳定梯度
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # 正则化， 随机丢弃10%的神经元，增强模型泛化能力
            nn.Conv2d(inter_channels, out_channels, 1)   # 1×1卷积层：通道数映射到类别数
        )

    def forward(self, x):
        """
         x:  例如：(1, 512, 16, 16)
        返回: 例如：(1, 21, 16, 16)，每个位置有21个类别的得分
        """
        return self.block(x)

class FCN32s(nn.Module):
    def __init__(self, nclass, backbone='vgg16', aux=False, pretrained_base=True,
                 norm_layer=nn.BatchNorm2d, **kwargs):

        super(FCN32s, self).__init__()
        self.aux = aux

        # 只加载VGG16的卷积部分（去除全连接层）
        if backbone == 'vgg16':
            self.pretrained = vgg16(pretrained=pretrained_base).features
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        # 分割头（1x1卷积）
        self.head = _FCNHead(512, nclass, norm_layer=norm_layer)

        if aux:
            self.auxlayer = _FCNHead(512, nclass, norm_layer=norm_layer)
        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

    def forward(self, x):
        # ([1, 3, 224, 224])
        size = x.size()[2:]

        # VGG16特征提取器（固定权重） conv1~conv5 下采样32倍
        pool5 = self.pretrained(x)  # ([1, 512, 7, 7])
        outputs = []

        # 分割头将其分割到目标的类别数
        out = self.head(pool5)      # ([1, 21, 7, 7])

        # 上采样32倍
        out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)   # ([1, 21, 224, 224])
        outputs.append(out)

        if self.aux:
            auxout = self.auxlayer(pool5)
            auxout = F.interpolate(auxout, size=size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)

class FCN16s(nn.Module):
    def __init__(self, nclass, backbone='vgg16', aux=False, pretrained_base=True,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCN16s, self).__init__()
        self.aux = aux
        if backbone == 'vgg16':

            self.pretrained = vgg16(pretrained=pretrained_base).features

        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.pool4 = nn.Sequential(*self.pretrained[:24])     # vgg 00~23层
        self.pool5 = nn.Sequential(*self.pretrained[24:])     # vgg 24~30层
        self.head = _FCNHead(512, nclass, norm_layer=norm_layer)
        self.score_pool4 = nn.Conv2d(512, nclass, 1)   # 跳跃连接模块
        if aux:
            self.auxlayer = _FCNHead(512, nclass, norm_layer=norm_layer)
        self.__setattr__('exclusive', ['head', 'score_pool4', 'auxlayer'] if aux else ['head', 'score_pool4'])

    def forward(self, x):
        # input size 1 3 224 224
        # vgg 特征提取
        pool4 = self.pool4(x)       # (_, 512, 14, 14)   ← 下采样16倍， 较浅特征，细节丰富
        pool5 = self.pool5(pool4)   # (_, 512, 7, 7)   ← 下采样32倍， 深层特征，语义丰富

        # 分别进行类别预测
        outputs = []
        score_fr = self.head(pool5)            # (nclass, 7, 7) 深层预测
        score_pool4 = self.score_pool4(pool4)  # (nclass, 14, 14) 浅层预测

        # 第一次上采样：深层特征上采样2倍到浅层分辨率
        upscore2 = F.interpolate(score_fr, score_pool4.size()[2:], mode='bilinear', align_corners=True)   # (_, nclass, 14, 14)

        # 特征融合：深层语义 + 浅层细节
        fuse_pool4 = upscore2 + score_pool4  # (_, nclass, 14, 14)

        # 最终上采样：16倍到原图尺寸
        out = F.interpolate(fuse_pool4, x.size()[2:], mode='bilinear', align_corners=True)  # (_, nclass, 224, 224)

        outputs.append(out)

        if self.aux:
            auxout = self.auxlayer(pool5)
            auxout = F.interpolate(auxout, x.size()[2:], mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class FCN8s(nn.Module):
    """
        1. 融合pool3(8倍下采样)、pool4(16倍下采样)、pool5(32倍下采样)三个层次的特征
        2. 两次跳跃连接，逐步上采样，获得精细的分割结果
        3. 输出相对于输入下采样8倍(8s的含义)
    """

    def __init__(self, nclass, backbone='vgg16', aux=False, pretrained_base=True,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCN8s, self).__init__()
        self.aux = aux  # 是否使用辅助损失

        # 加载并拆分VGG16主干网络
        if backbone == 'vgg16':
            self.pretrained = vgg16(pretrained=pretrained_base).features # 只使用VGG的卷积部分(features)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        # 将VGG的卷积层拆分为三个部分，对应三个下采样级别， VGG16 features共有31层(0-30)
        self.pool3 = nn.Sequential(*self.pretrained[:17])    # 层0-16: Conv1_1 ~ Conv3_3 (第3个池化层前)
        self.pool4 = nn.Sequential(*self.pretrained[17:24])  # 层17-23: Conv4_1 ~ Conv4_3 (第4个池化层前)
        self.pool5 = nn.Sequential(*self.pretrained[24:])    # 层24-30: Conv5_1 ~ Conv5_3+MaxPool5

        # 分割头
        self.head = _FCNHead(512, nclass, norm_layer)

        # 跳跃连接的1x1卷积 - 调整浅层特征的通道数
        self.score_pool3 = nn.Conv2d(256, nclass, 1)  # pool3输出256通道
        self.score_pool4 = nn.Conv2d(512, nclass, 1)  # pool4输出512通道

        # 辅助输出头(可选) - 用于训练时的辅助监督
        if aux:
            self.auxlayer = _FCNHead(512, nclass, norm_layer)

        # 标识需要特殊训练处理的模块，可设定不同的学习率
        self.__setattr__('exclusive',
                         ['head', 'score_pool3', 'score_pool4', 'auxlayer'] if aux
                         else ['head', 'score_pool3', 'score_pool4'])

    def forward(self, x):
        """
        x:   输入图像，形状为(batch_size, 3, H, W)
        返回: 分割结果，形状为(batch_size, nclass, H, W)
        """
        # ========== 阶段1：特征提取 ==========
        pool3 = self.pool3(x)      # 提取pool3特征 (下采样8倍)  → (batch, 256, H/8, W/8)
        pool4 = self.pool4(pool3)  # 提取pool4特征 (下采样16倍) → (batch, 512, H/16, W/16)
        pool5 = self.pool5(pool4)  # 提取pool5特征 (下采样32倍) → (batch, 512, H/32, W/32)

        outputs = []  # 存储所有输出，主要方便查看中间结果，可以没有
        # ========== 阶段2：三个层次的特征预测 ==========
        score_fr = self.head(pool5)            # 深层特征预测 → (batch, nclass, H/32, W/32)
        score_pool4 = self.score_pool4(pool4)  # 中层特征预测 → (batch, nclass, H/16, W/16)
        score_pool3 = self.score_pool3(pool3)  # 浅层特征预测 → (batch, nclass, H/8, W/8)

        # ========== 阶段3：第一次融合 (pool5 + pool4) ==========
        # 将深层特征上采样2倍到pool4的分辨率， 深层语义 + 中层细节
        upscore2 = F.interpolate(score_fr, score_pool4.size()[2:], mode='bilinear', align_corners=True)
        fuse_pool4 = upscore2 + score_pool4    # → (batch, nclass, H/16, W/16)

        # ========== 阶段4：第二次融合 (fuse_pool4 + pool3) ==========
        # 将融合后的特征上采样2倍到pool3的分辨率， 中层融合特征 + 浅层细节
        upscore_pool4 = F.interpolate(fuse_pool4, score_pool3.size()[2:],  mode='bilinear', align_corners=True)
        fuse_pool3 = upscore_pool4 + score_pool3  # → (batch, nclass, H/8, W/8)

        # ========== 阶段5：最终上采样 ==========
        # 将融合了三个层次的特征上采样8倍到原始输入尺寸
        out = F.interpolate(fuse_pool3, x.size()[2:], mode='bilinear', align_corners=True)
        outputs.append(out)

        # ========== 阶段6：辅助输出 (可选) ==========
        if self.aux:
            # 直接从pool5特征预测并上采样32倍
            auxout = self.auxlayer(pool5)
            auxout = F.interpolate(auxout, x.size()[2:], mode='bilinear', align_corners=True)
            outputs.append(auxout)  # 辅助输出
        return tuple(outputs)  # 返回元组: (主输出, 辅助输出) 或 (主输出,)

if __name__ == '__main__':
    from thop import profile
    device = torch.device('cpu')
    model = FCN8s(21)
    # model = FCN16s(21)
    # model = FCN32s(21)
    model = model.to(device)

    input = torch.randn(1, 3, 224, 224)
    input = input.to(device)

    flops, params = profile(model, inputs=(input, ))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
