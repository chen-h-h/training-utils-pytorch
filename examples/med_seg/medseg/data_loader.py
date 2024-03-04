import os
import torch
import random
import numpy as np
import os.path as osp
from glob import glob
from PIL import Image, ImageChops
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import logging
logger = logging.getLogger(__name__)

'''Transforms for PIL.Image(W*H)'''

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        
        return img, mask
    
class RandomRotate(object):
    def __init__(self, angle=45, prob=0.5):
        self.angle = angle
        self.prob = prob

    def __call__(self, img, mask):
        assert img.size == mask.size
        if random.random() < self.prob:
            rotate_angle = random.randint(-self.angle, self.angle)
            img = img.rotate(rotate_angle)
            mask = mask.rotate(rotate_angle)
        
        return img, mask
    
class RandomOffset(object):
    def __init__(self, offset_scale=0.2,prob=0.5):
        self.scale = offset_scale
        self.prob = prob

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        self.woffset, self.hoffset = int(self.scale * w), int(self.scale * h)
        
        if random.random() < self.prob:
            xoffset = random.randint(-self.woffset, self.woffset)
            yoffset = random.randint(-self.hoffset, self.hoffset)
            
            img = ImageChops.offset(img,xoffset,yoffset)
            mask = ImageChops.offset(mask,xoffset,yoffset)
        
        return img, mask

class Normalization(object):
    def __init__(self):
        pass

    def __call__(self, img, mask):
        assert img.size == mask.size
        img = np.array(img)
        img = img/255
        img = (img-0.5)/0.5 # convert to [-1, 1]
        img = Image.fromarray(img)

        return img, mask

class Med_Dataset(Dataset):
    """
    return:
        input:img(CHW)
        target:mask(HW)
    """
    def __init__(self, data_path, transform=None):
        self.data_list = glob(osp.join(data_path,'Normal/images/*.png'))
        self.transform  = transform
        logger.info(f'Creating dataset with {len(self.data_list)} examples')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path = self.data_list[idx]
        mask_path = img_path.replace('Normal/images','Normal/lung masks')
        img = Image.open(img_path)    # W*H
        mask = Image.open(mask_path)  # W*H
      
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        img = torch.from_numpy(np.array(img)[None].astype(np.float32))
        mask = torch.from_numpy(np.array(mask).astype(np.float32))
        mask[mask==255]=1.0     # Labels must be continuous values starting from zero

        return {'input': img, 'target': mask}

def get_data_loader(is_distributed, data_root, batch_size):
        
    train_transform = Compose([
                            RandomOffset(offset_scale=0.2,prob=0.5),
                            RandomRotate(angle=45,prob=0.5),
                            Normalization(),
                        ])
    test_transform = Compose([Normalization()])
  
    train_dataset = Med_Dataset(data_path=osp.join(data_root,'Train'), transform=train_transform)
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                            shuffle=(train_sampler is None), num_workers=4)
    
    test_dataset = Med_Dataset(data_path=osp.join(data_root,'Test'), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader