import os
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms as standard_transforms
import numpy as np

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''

def make_dataset(mode,  root, max_files=None):
    assert mode in ['train', 'val', 'test']
    items = []
    img_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
    mask_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'SegmentationClass')
    data_list = [l.strip('\n') for l in open(os.path.join(
        root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', '{}.txt'.format(mode))).readlines()]
    for it in data_list:
        if mode != 'test':
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
        else:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'), it)
        items.append(item)
    if max_files is not None:
        items = items[:max_files]
    print('dataset size for {}: {}'.format(mode, len(items)))
    return items

def Palette():
    return [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]  #3 values- R,G,B for every class. First 3 values for class 0, next 3 for
#class 1 and so on......

class VOC(data.Dataset):
    '''
    transform为输入的标准化、格式转变等处理函数，target_transform为标签的格式转变函数
    '''
    def __init__(self, mode,  root, resize_to, transform=None, target_transform=None, ignore_label=255, max_files=None):
        self.imgs = make_dataset(mode,  root, max_files)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.width = resize_to[0]
        self.height = resize_to[1]
        self.ignore_label = ignore_label

    def __getitem__(self, index):

        if self.mode != 'test':
            img_path, mask_path = self.imgs[index]
        else:
            img_path, mask_path, img_name = self.imgs[index]

        img = Image.open(img_path).convert('RGB').resize((self.width, self.height))
        mask = Image.open(mask_path).resize((self.width, self.height))    # 模式为'P'(调色板),调色板索引[0-20, 255]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        mask = torch.round(mask)
        mask[mask==self.ignore_label]=0   # 边界处理 255→0

        if self.mode != 'test':
            return img, mask
        else:
            return img, mask, img_name

    def __len__(self):
        return len(self.imgs)

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

class MaskToTensor_input(object):
    def __init__(self, resize_to):
        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),  # PIL→Tensor, [0,255]→[0,1], (H,W,C)→(C,H,W)
            standard_transforms.RandomHorizontalFlip(),  # 随机水平翻转 p=0.5
            standard_transforms.RandomVerticalFlip(),  # 随机垂直翻转 p=0.5
            standard_transforms.RandomAffine((5, 10)),  # 随机仿射变换 (±5-10度旋转)
            standard_transforms.RandomCrop(resize_to, padding=2),  # 随机裁剪到220×220
            standard_transforms.Normalize(*self.mean_std)  # 标准化: (x-mean)/std
        ])
    def __call__(self, img):
        res = self.input_transform(img)
        return res

if __name__ == '__main__':
    ignore_label = 255
    root = 'F:/ImageSegment'
    resize_to = (224,224)

    input_transform = MaskToTensor_input(resize_to=(224,224))
    target_transform = MaskToTensor()

    # train_dataset = VOC('train', root, resize_to, transform=input_transform, target_transform=target_transform, ignore_label=255)
    # val_dataset = VOC('val',  root, resize_to, transform=input_transform, target_transform=target_transform)
    test_dataset = VOC('test',  root, resize_to, transform=input_transform, target_transform=target_transform)

    # train_loader = data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    # val_loader = data.DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    for batch, data in enumerate(test_loader):
        x, y, z = data
        print(y.max(), y.min())
        print(z)
        print(x.shape)
        print(y.shape)
        exit()