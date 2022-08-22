# YOLOv5 Oneflow
## Introduction
This is the oneflow reimplementation of the repo: [yolov5](https://github.com/ultralytics/yolov5)
## Prerequisites
```
pip3 install -r requirements.txt
```
## Getting start
### Dataset
The MS COCO dataset is a large-scale object detection, segmentation, and captioning dataset published by Microsoft. Machine Learning and Computer Vision engineers popularly use the COCO dataset for various computer vision projects.

'data/coco.yaml' has the place of dataset. and if the dataset is missed, it will autodownload the whole coco dataset.

You can change below code to change dataset or use your custom dataset by change the .yaml file.
```
parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='dataset.yaml path')
```
### Pretrained model
Please get the Pretrained model from:


the pretrained model is from [yolov5](https://github.com/ultralytics/yolov5) which is converted to oneflow model.

### Training 
To train yolov5 models from sctrach on coco dataset:
```
python train.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128
                                       yolov5s                                64
                                       yolov5m                                40
                                       yolov5l                                24
                                       yolov5x                                16
```
more parameters can be viewed at train.py.
To use DDP:
```
python3 -m oneflow.distributed.launch --nproc_per_node 2 ./train.py
```
### Val
To val yolov5 models perfermance:
```
python val.py --data coco.yaml --weights 'ckpt' --batch-size 128
```
more parameters can be viewed at val.py.
### Inference
To inference on single picture:
```
python detect.py --img img.jpg
```

All results can be found in run/(train)/(val)/(detect)/

