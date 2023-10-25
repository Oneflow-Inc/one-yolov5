python train.py --weights yolov5s.of --device 0 


python train.py --resume runs/train/exp/weights/last.of


python val.py --weights yolov5x.of --data data/coco128.yaml --img 640 --half


python detect.py --weights yolov5s.of --img 832 --source data/images/ --conf-thres 0.8