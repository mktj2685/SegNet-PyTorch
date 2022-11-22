import os.path as osp
import random
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms


class VOC2012(Dataset):

    def __init__(self, phase:str):
        self.fnames = []
        with open(f'data/VOC2012/ImageSets/Segmentation/{phase}.txt', 'r') as f:    # phase is 'train' or 'val'
            self.fnames = [line.rstrip() for line in f.readlines()]
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        img = Image.open(osp.join('data/VOC2012/JPEGImages', f'{self.fnames[index]}.jpg')).convert('RGB')       # RGB image
        mask = Image.open(osp.join('data/VOC2012/SegmentationClass', f'{self.fnames[index]}.png')).convert('P') # indexed image
        # NOTE Since img and mask must be flipped together, transforms.RandomHorizontalFlip is not available.
        if random.random() < 0.5:
            img = F.hflip(img)
            mask = F.hflip(mask)
        img = F.resize(img, (415, 415))
        mask = F.resize(mask, (415, 415))
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = F.pil_to_tensor(mask).long()
        return img, mask


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = VOC2012('train')
    dataloader = DataLoader(dataset, 1, False)
    img, mask = next(iter(dataloader))
    from utils.color_pallet import COLOR_PALLET
    mask = Image.fromarray(mask[0, 0].to(torch.uint8).cpu().numpy(), 'P')
    mask.putpalette(COLOR_PALLET, rawmode='RGB')
    mask = mask.convert('RGB')
    import numpy as np
    print(np.array(mask))
    mask.show()