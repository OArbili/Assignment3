import os
import time
import timm
import torch
import config
import numpy as np
import pandas as pd
from PIL import Image

import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from tqdm.notebook import tqdm
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision import transforms

from preprocess import Flower102Data,GradualWarmupSchedulerV2

''' This is the main class which runs the pre-trained CNN resnet101 and EfficientNet_b1_ns, efficientnet_b3, efficientnet_b4_ns'''
class part_2_cnn():
    
    def __init__(self,in_model_type):
        # load config
        self.cfg = config.model_cfg(in_model_type)

        # data objects
        self.dataset_train = None
        self.dataset_test = None
        self.dataset_valid = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        
        #transformation objects
        self.normalize = None
        self.train_trns = None
        self.test_trns = None
        
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = amp.GradScaler()
    
    ''' This function in charge of the image transofrmation. I used the following transofrmations
    - RandomResizedCrop
    - RandomHorizontalFlip
    - RandomRotation
    - GaussianBlur
    '''
    def train_test_img_transformation(self):
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        self.train_trns = transforms.Compose([
            transforms.RandomResizedCrop(self.cfg.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(360),
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.test_trns = transforms.Compose([
            transforms.Resize((self.cfg.image_size,self.cfg.image_size)),
            transforms.ToTensor(),
            self.normalize,
        ])
    
    ''' This function load the data using the helper class Flower102Data in preprocess file '''
    def load_data(self):
        self.dataset_train = Flower102Data(self.cfg.root,self.cfg.ids_train, True, self.train_trns, shots=-1, seed=self.cfg.seed)
        self.dataset_test = Flower102Data(self.cfg.root,self.cfg.ids_test, False, self.test_trns, shots=-1, seed=self.cfg.seed)
        self.dataset_valid = Flower102Data(self.cfg.root,self.cfg.ids_val, False, self.test_trns, shots=-1, seed=self.cfg.seed)
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset_valid, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)
    
    def train_epoch(self,model, loader, optimizer):

        model.train()
        train_loss = []
        LOGITS = []
        TARGETS = []
        bar = tqdm(loader)
        for (data, targets) in bar:

            optimizer.zero_grad()
            data, targets = data.cuda(), targets.cuda()

            with amp.autocast():
                logits = model(data)
                loss = self.criterion(logits, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            loss_np = loss.item()
            train_loss.append(loss_np)
            LOGITS.append(logits.detach().cpu())
            TARGETS.append(targets.detach().cpu())
            smooth_loss = sum(train_loss[-50:]) / min(len(train_loss), 50)
            bar.set_description('loss: %.4f, smth: %.4f' % (loss_np, smooth_loss))

        train_loss = np.mean(train_loss)
        LOGITS = torch.cat(LOGITS).float()
        LOGITS = LOGITS.softmax(1).cpu().numpy()
        TARGETS = torch.cat(TARGETS).cpu().numpy()
        acc = accuracy_score(LOGITS.argmax(1),TARGETS)
        return train_loss,acc


    def valid_epoch(self,model, loader, get_output=False):

        model.eval()
        val_loss = []
        LOGITS = []
        TARGETS = []
        with torch.no_grad():
            for (data, targets) in tqdm(loader):
                data, targets = data.cuda(), targets.cuda()
                logits = model(data)
                loss = self.criterion(logits, targets)
                val_loss.append(loss.item())
                LOGITS.append(logits.cpu())
                TARGETS.append(targets.cpu())

        val_loss = np.mean(val_loss)
        LOGITS = torch.cat(LOGITS)
        LOGITS = LOGITS.softmax(1).cpu().numpy()
        TARGETS = torch.cat(TARGETS).cpu().numpy()
        acc = accuracy_score(LOGITS.argmax(1),TARGETS)
        return val_loss, acc
    
    ''' This function call the pre-trained model '''
    def build_model(self,fold=0):
        res_df = pd.DataFrame(columns=['Epoch','lr','train_loss','valid_loss','train_acc','valid_acc','test_loss','score'])
        self.model = timm.create_model(self.cfg.enet_type, pretrained=True,num_classes=self.cfg.num_classes).cuda()
        aucs_max = 0
        self.model_file = os.path.join(self.cfg.model_dir, f'{self.cfg.kernel_type}_best_fold{fold}.pth')
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.init_lr)
        self.scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.cfg.cosine_epo)
        self.scheduler_warmup = GradualWarmupSchedulerV2(self.optimizer, multiplier=3, total_epoch=self.cfg.warmup_epo, after_scheduler=self.scheduler_cosine)
        for epoch in range(1, self.cfg.n_epochs+1):
            print(time.ctime(), 'Epoch:', epoch)
            self.scheduler_warmup.step(epoch-1)

            train_loss,train_acc = self.train_epoch(self.model, self.train_loader, self.optimizer)
            val_loss, aucs = self.valid_epoch(self.model, self.valid_loader)

            content = time.ctime() + ' ' + f'Fold {fold} Epoch {epoch}, lr: {self.optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.4f}, valid loss: {(val_loss):.4f}, train_acc: {train_acc:.4f}, valid_acc: {aucs:.4f}.'
            res_df = res_df.append({'Epoch':epoch,
                                    'lr':self.optimizer.param_groups[0]["lr"],
                                    'train_loss':train_loss,
                                    'valid_loss':val_loss,
                                    'train_acc':train_acc,
                                    'valid_acc':aucs,
                                   },ignore_index=True)
            print(content)
            with open(self.cfg.log_file, 'a') as appender:
                appender.write(content + '\n')

            if aucs_max < np.mean(aucs):
                print('acc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(aucs_max, np.mean(aucs)))
                torch.save(self.model.state_dict(), self.model_file)
                aucs_max = np.mean(aucs)
        self.model.load_state_dict(torch.load(self.model_file), strict=True)
        acc_test = self.valid_epoch(self.model, self.test_loader)
        print(f"test loss {acc_test[0]} score {acc_test[1]}")
        with open(self.cfg.log_file, 'a') as appender:
            appender.write(f"test loss {acc_test[0]} score {acc_test[1]}" + '\n')
        # save epocsresults to csv
        res_df['test_loss'] = acc_test[0]
        res_df['score'] = acc_test[1]
        res_df.to_csv(self.cfg.res_df_loc);