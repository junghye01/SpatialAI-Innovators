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

#from unet3plus import UNet3Plus
from dataset import SatelliteDataset,CustomDataset
from utils import calculate_acc,calculate_iou,DiceLoss_customized

import wandb

def save_model_state_dict(model,saved_dir,file_name='ensembleUNet_best_model_dsv.pt'):
    output_path=os.path.join(saved_dir,file_name)
    torch.save(model.state_dict(),output_path)

def train_ensemble(num_epochs,models,data_loader,val_loader,criterion,optimizer,scheduler,val_every,saved_dir,device):
   
    min_loss=100.0

    for epoch in range(num_epochs):
        print(f'Start train #{epoch+1}')
        
        for model in models:
            model.train()
      
        accuracy_list=[]
        iou_list=[]
        epoch_loss=0

        for images,masks,_ in tqdm(data_loader):
            images=images.float().to(device)
            masks=masks.float().to(device)

            for model in models:

                optimizer.zero_grad()
                outputs=model(images)
        
                # pred와 true
                masks_pred=torch.sigmoid(outputs).detach().cpu().numpy()
                masks_true=masks.detach().cpu().numpy()
                
            
                masks_pred = np.squeeze(masks_pred, axis=1)
            #masks_true=np.squeeze(masks_true,axis=1)
            #print(masks_pred.shape,masks_true.shape)
                for i in range(len(images)):

                    accuracy=calculate_acc(masks_pred[i],masks_true[i],threshold=0.35)
                    accuracy_list.append(accuracy)

                    iou=calculate_iou(masks_pred[i],masks_true[i],threshold=0.35)
                    iou_list.append(iou)

            
                loss = criterion(outputs, masks.unsqueeze(1))
                loss.backward()
                optimizer.step()

                epoch_loss+=loss.item()

        # epoch마다 acc, iou
        mean_accuracy=np.mean(accuracy_list)
        mean_iou=np.mean(iou_list)

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/(len(models)*len(data_loader)):.3f},acc:{mean_accuracy:.3f},iou:{mean_iou:.3f}')

        # validation 주기에 따른 loss 출력 
        
        if (epoch+1)%val_every==0:
            val_loss,val_acc,val_iou=val_ensemble(epoch+1,models,val_loader,criterion,device)
            if val_loss<min_loss:
                print(f'Best performance at epoch : {epoch+1}')
                print(f'Save model in {saved_dir}')
                min_loss=val_loss
                # model parameter save
                #save_model_state_dict(model,saved_dir)
                for i, model in enumerate(models):
                    output_path = os.path.join(saved_dir, f'ensembleUNet_model_{i}.pt')
                    torch.save(model.state_dict(), output_path)

            wandb.log({
                "valid_loss":round(val_loss,3),
                "valid_acc": round(val_acc,3),
                "valid_iou":round(val_iou,3),

                
            })
        # 매 에폭마다 스케줄러 호출
        scheduler.step()
            
        
        # wandb
        wandb.log({
            "learning_rate": optimizer.param_groups[0]['lr'],
            "train_loss":round(epoch_loss/(len(models)*len(data_loader)),3),
            "train_miou":round(mean_iou,3),
            #"val_loss":round(val_loss,3),
            #"val_acc":round(val_acc,3),
            #"val_iou":round(val_iou,3),

        })
        

            

def val_ensemble(epoch,models,data_loader,criterion,device):
    print(f'Start validation #{epoch}')

    for model in models:
        model.eval().to(device)

    accuracy_list=[]
    iou_list=[]
    val_loss=0
    

    with torch.no_grad():
        for images,masks,_ in tqdm(data_loader):
            images=images.float().to(device)
            masks=masks.float().to(device)

            # 각 model 마다 output 계산
            outputs_list=[]
            for model in models:
                outputs=model(images)
                outputs_list.append(outputs)

            combined_outputs=torch.mean(torch.stack(outputs_list,dim=0),dim=0)

            loss=criterion(combined_outputs,masks.unsqueeze(1))
            val_loss+=loss.item()

            # acc,iou
            masks_pred=torch.sigmoid(combined_outputs).detach().cpu().numpy()
            masks_true=masks.detach().cpu().numpy()

            for i in range(len(images)):

                accuracy=calculate_acc(masks_pred[i],masks_true[i],threshold=0.35)
                accuracy_list.append(accuracy)

                iou=calculate_iou(masks_pred[i],masks_true[i],threshold=0.35)
                iou_list.append(iou)

            

    mean_accuracy=np.mean(accuracy_list)
    mean_iou=np.mean(iou_list)
    val_loss=val_loss/(len(models)*len(data_loader))
    print(f'Val #{epoch}, Loss: {val_loss:.3f},acc:{mean_accuracy:.3f},iou:{mean_iou:.3f}')
    
    return val_loss,mean_accuracy,mean_iou

    



if __name__=='__main__':

    wandb.init(project='semantic-segmentation',name='EnsembleUNet_dsv')
    
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

    #test_transform=A.Compose([
     #   A.Lambda(image=preprocessing_fn),
      #  ToTensorV2()
    #])


    train_dataset = CustomDataset(csv_file='./data/new_train.csv',mode='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    val_dataset = CustomDataset(csv_file='./data/new_val.csv',mode='val', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Define ensemble model
    criterion = DiceLoss_customized()

    models=[]
    num_models=3
    # parameter
    num_epochs=12
    val_every=3
    saved_dir='/home/irteam/junghye-dcloud-dir/SpatialAI-Innovators/models'
  

    for i in range(num_models):
        model = smp.Unet(classes=1).to(device)
        wandb.watch(model)
        models.append(model)
    

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=5 * 1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-7)

    train_ensemble(num_epochs,models,train_loader,val_loader,criterion,optimizer,scheduler,val_every,saved_dir,device)
    # 모델 매개변수만 저장
    #output_path='/home/irteam/junghye-dcloud-dir/SpatialAI-Innovators/data/UNet3Plus_0727.pt'
    #torch.save(model.state_dict(),output_path)
