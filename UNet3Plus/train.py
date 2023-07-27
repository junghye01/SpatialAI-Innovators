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

from unet3plus import UNet3Plus
from dataset import SatelliteDataset
from utils import *

if __name__=='__main__':
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn('efficientnet-b5','imagenet')

    train_transform = A.Compose(
        [   
            A.Resize(512,512),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.Lambda(image=preprocessing_fn),
            ToTensorV2()
        ]
    )

    test_transform=A.Compose([
        A.Lambda(image=preprocessing_fn),
        ToTensorV2()
    ])


    train_dataset = SatelliteDataset(csv_file='./data/train.csv', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)


    #test_dataset = SatelliteDataset(csv_file='./data/test.csv', transform=test_transform, infer=True)
    #test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

        # model 초기화
    model = UNet3Plus().to(device)

    # loss function과 optimizer 정의
    criterion = DiceLoss_customized()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=5 * 1e-5, weight_decay=0.01)

    
    for epoch in range(10):
        model.train()
        accuracy_list=[]
        iou_list=[]
        epoch_loss=0

        for images,masks in tqdm(train_loader):
            images=images.float().to(device)
            masks=masks.float().to(device)

            optimizer.zero_grad()
            outputs=model(images)
        
            # pred와 true
            masks_pred=torch.sigmoid(outputs).detach().cpu().numpy()
            masks_true=masks.detach().cpu().numpy()

            
            masks_pred=np.squeeze(masks_pred,axis=1)
            #masks_true=np.squeeze(masks_true,axis=1)
            #print(masks_pred.shape,masks_true.shape)
            for i in range(len(images)):

                accuracy=calculate_acc(masks_pred[i],masks_true[i],threshold=0.35)
                accuracy_list.append(accuracy)

                iou=calculate_iou(masks_pred[i],masks_true[i],threshold=0.35)
                iou_list.append(iou)

            
            loss=criterion(outputs,masks.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss+=loss.item()

        # epoch마다 acc, iou
        mean_accuracy=np.mean(accuracy_list)
        mean_iou=np.mean(iou_list)

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)},acc:{mean_accuracy:.3f},iou:{mean_iou:.3f}')


    # 모델 매개변수만 저장
    output_path='/home/irteam/junghye-dcloud-dir/SpatialAI-Innovators/data/UNet3Plus_0727.pt'
    torch.save(model.state_dict(),output_path)
