# SpatialAI-Innovators
위성 이미지 건물 영역 분할

* new_unnetthreeplus.py : dataset 클래스, dataloader 정의 모두 있기 때문에 그대로 실행하면 되는 스크립트 (unet baseline에서 조금 수정해서 unet3+ 만들었음) , train 해서 모델 저장하는 거까지 test 코드는 x

* train.ipynb : unetthreeplus.py 에서 정의한 unet3+ 로 train , test 시키는 코드

* utils.py : accuracy , IoU 점수 정의

* satellitedataset.py : train.ipynb 에서 dataset 클래스 불러올 때 쓰는 모듈