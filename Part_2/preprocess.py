import os
import numpy as np
from PIL import Image
import pandas as pd
import scipy.io as sio
import torch.utils.data as data
from warmup_scheduler import GradualWarmupScheduler

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

class Flower102Data(data.Dataset):
    
    def __init__(self, root ,ids , is_train=True, transform=None, shots=-1, seed=0, preload=False,num_classes = 102):
        self.preload = preload
        self.num_classes = num_classes
        self.transform = transform
        imglabel_map = os.path.join(root, 'imagelabels.mat')
        setid_map = os.path.join(root, 'setid.mat')


        imagelabels = sio.loadmat(imglabel_map)['labels'][0]
        setids = sio.loadmat(setid_map)

        self.labels = []
        self.image_path = []

        for i in ids:
            # Original label start from 1, we shift it to 0
            self.labels.append(int(imagelabels[i-1])-1)
            self.image_path.append( os.path.join(root, 'jpg', 'image_{:05d}.jpg'.format(i)) )

        self.labels = np.array(self.labels)

        new_img_path = []
        new_img_labels = []
        if is_train:
            if shots != -1:
                self.image_path = np.array(self.image_path)
                for c in range(self.num_classes):
                    ids = np.where(self.labels == c)[0]
                    random.seed(seed)
                    random.shuffle(ids)
                    count = 0
                    new_img_path.extend(self.image_path[ids[:shots]])
                    new_img_labels.extend([c for i in range(shots)])
                self.image_path = new_img_path
                self.labels = new_img_labels

        if self.preload:
            self.imgs = {}
            for idx in range(len(self.image_path)):
                if idx % 100 == 0:
                    print('Loading {}/{}...'.format(idx+1, len(self.image_path)))
                img = Image.open(self.image_path[idx]).convert('RGB')
                self.imgs[idx] = img

    def __getitem__(self, index):
        if self.preload:
            img = self.imgs[index]
        else:
            img = Image.open(self.image_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, self.labels[index]

    def __len__(self):
        return len(self.labels)