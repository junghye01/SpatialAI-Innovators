# SpatialAI-Innovators
위성 이미지 건물 영역 분할

### Dice Loss 기준으로 Best Model 선정 

## EnsembleUNet (segmentation_models_pytorch에서 제공하는 Unet 모델)
### Best Model (Dice Loss, acc, IoU): 0.167, 0.951, 0.337
### K-Fold Cross Validation 
### Best Model (Dice Loss, acc , IoU): 0.064 , 0.977 , 0.665

## DeepLabV3
### Best Model (Dice Loss, acc, IoU) :0.424, 0.952, 0.406


## UNet3Plus (segmentation_models_pytorch 에서 제공하는 Unet3plus 모델)
### K-Fold Cross Validation
### Best Model (Dice Loss, acc, IoU) : 0.071 , 0.975, 0.639

