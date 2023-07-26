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


# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        img_path=img_path.replace('./','/home/irteam/junghye-dcloud-dir/SpatialAI-Innovators/data/')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

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




test_dataset = SatelliteDataset(csv_file='./data/test.csv', transform=test_transform, infer=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)


# U-Net의 기본 구성 요소인 Double Convolution Block을 정의합니다.
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# 간단한 U-Net3+ 모델 정의
class UNet3Plus(nn.Module):
    def __init__(self):
        super(UNet3Plus, self).__init__()
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.dconv_down5 = double_conv(512, 1024)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

        self.dconv_up4 = double_conv(1024 + 512, 512)
        self.dconv_up3 = double_conv(512 + 256, 256)
        self.dconv_up2 = double_conv(256 + 128, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)  

        x = self.dconv_down5(x)

        x = self.upsample(x)        
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)

        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out

# 모델 생성 및 확인
model = UNet3Plus()

class DiceLoss_customized(nn.Module):
    def __init__(self):
        super(DiceLoss_customized, self).__init__()

    def forward(self, inputs, targets, smooth=1e-7):
        
        inputs = torch.sigmoid(inputs) # sigmoid를 통과한 출력이면 주석처리
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

# model 초기화
model = UNet3Plus().to(device)

# loss function과 optimizer 정의
criterion = DiceLoss_customized()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=5 * 1e-5, weight_decay=0.01)

for epoch in range(10):
    model.train()
    epoch_loss=0
    for images,masks in tqdm(train_loader):
        images=images.float().to(device)
        masks=masks.float().to(device)

        optimizer.zero_grad()
        outputs=model(images)
       # print(images.shape,outputs.shape)
        
        loss=criterion(outputs,masks.unsqueeze(1))
        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()

    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}')


output_path='/home/irteam/junghye-dcloud-dir/SpatialAI-Innovators/data/unet3plus_new.pt'
torch.save(model,output_path)