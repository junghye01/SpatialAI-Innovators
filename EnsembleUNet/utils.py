# # https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import torch.nn as nn
import torch


def calculate_acc(predicted_masks,true_masks,threshold=0.35):
    # predicted masks와 true masks가 numpy 배열일 때
    predicted_masks=(predicted_masks>threshold).astype(np.uint8)
    true_masks=true_masks.astype(np.uint8)

    correct_pixels=np.sum(predicted_masks==true_masks)

    total_pixels=true_masks.size

    accuracy=correct_pixels/total_pixels
    return accuracy

def calculate_iou(predicted_masks,true_masks,threshold=0.35):
    predicted_masks=(predicted_masks>threshold).astype(np.uint8)
    true_masks=true_masks.astype(np.uint8)

    intersection=np.sum(predicted_masks & true_masks)
    union=np.sum(predicted_masks | true_masks)

    iou=intersection/union
    return iou


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


