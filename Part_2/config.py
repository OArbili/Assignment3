import os
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split

''' This config class holds all model configurations. I used similar configuration for all pre-trained networks'''
class model_cfg():
    
    def __init__(self,in_model_type):
        
        assert(in_model_type in ['resnet101' ,'tf_efficientnet_b1_ns' , 'tf_efficientnet_b4_ns' , 'efficientnet_b3'])
        
        self.DEBUG = False
        self.enet_type = in_model_type
        self.kernel_type = f'train_{self.enet_type}'

        self.root = './data'
        self.num_workers = 6
        self.num_classes = 102
        # n_ch = 3
        self.image_size = 256
        self.batch_size = 64
        self.init_lr = 3e-4
        self.warmup_epo = 1

        self.cosine_epo = 19 if not self.DEBUG else 2
        self.n_epochs = self.warmup_epo + self.cosine_epo

        self.log_dir = './results'
        self.model_dir = './models'
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f'res_{self.kernel_type}.txt')
        self.res_df_loc = os.path.join(self.log_dir, f'res_{self.kernel_type}_df.csv')
        self.seed= int(101)
        self.imglabel_map = os.path.join(self.root, 'imagelabels.mat')
        self.setid_map = os.path.join(self.root, 'setid.mat')
        self.imagelabels = sio.loadmat(self.imglabel_map)['labels'][0]
        self.setids = sio.loadmat(self.setid_map)
        self.ids = np.concatenate([self.setids['trnid'][0], self.setids['valid'][0],self.setids['tstid'][0]])
        self.ids_train, self.ids_test, _, _ = train_test_split(self.ids, self.ids, test_size=0.5, random_state=42)
        self.ids_test, self.ids_val, _, _ = train_test_split(self.ids_test, self.ids_test, test_size=0.5, random_state=42)
        
            
