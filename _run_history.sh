python train.py --weights yolov5s.of --device 0 


python train.py --resume runs/train/exp/weights/last.of


python val.py --weights yolov5x.of --data data/coco128.yaml --img 640 --half


python detect.py --weights yolov5s.of --img 832 --source data/images/ --conf-thres 0.8


###################################### 2 ######################################################################


# step1: download the dataset

https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.2.0/snake_datasets.zip

mv snake_datasets ../datasets/

# step2: make your own dataset yaml file

data/snake_data.yaml


# step3: train
python train.py --data data/snake_data.yaml --weights yolov5s.of 


# get the best model
runs/train/exp5/weights/best.of


python detect.py --weights runs/train/exp5/weights/best.of --source 236cc5a4d52b1f1c054ef5080e3c6e8a_w.jpeg


python detect.py --weights runs/train/exp5/weights/best.of --source 1126e121483c3d8e088cc96f548d5796_w.jpeg