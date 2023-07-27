import os
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dataset import SatelliteDataset,CustomDataset
from utils import calculate_acc,calculate_iou,DiceLoss_customized
from sklearn.model_selection import KFold
from train import train_ensemble,val_ensemble
import wandb


if __name__=='__main__':
    
    wandb.init(project='semantic-segmentation',name='EnsembleUNet_kfold_dsv')

    preprocessing_fn = smp.encoders.get_preprocessing_fn('efficientnet-b5','imagenet')

    train_transform = A.Compose(
        [   
            A.Resize(256,256),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.Lambda(image=preprocessing_fn),
            ToTensorV2()
        ]
    )

    val_transform=A.Compose(
        [
            A.Lambda(image=preprocessing_fn),
            ToTensorV2()
        ]
    )

 

    train_dataset = SatelliteDataset(csv_file='./data/train.csv', transform=train_transform)
    
    # parameter
   
    num_folds = 5
    num_epochs=24
    num_models=3
    val_every=12
    min_val_loss=100
    checkpoint=1
    saved_dir='/home/irteam/junghye-dcloud-dir/SpatialAI-Innovators/models'

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # K-Fold Cross Validation
    fold=0

    # model 
    models=[]

    for i in range(num_models):
        model=smp.Unet(classes=1).to(device)
        wandb.watch(model)
        models.append(model)

    optimizers=[]
    schedulers=[]

    for model in models:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=5 * 1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-7)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    for train_index,val_index in kf.split(train_dataset):
        fold+=1
        print(f'Fold {fold}')
        min_loss=min_val_loss

        # Create train and validation data for this fold
        train_data = torch.utils.data.Subset(train_dataset, train_index)
        val_data = torch.utils.data.Subset(train_dataset, val_index)
        
        # Create DataLoader for train and validation data
        train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=4)


        criterion=DiceLoss_customized()

        # train
        min_val_loss,checkpoint=train_ensemble(num_epochs,models,train_loader,val_loader,criterion,optimizers,schedulers,val_every,saved_dir,device,min_loss,checkpoint)

    print('K-Fold Cross Validation completed.')