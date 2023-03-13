#!/bin/bash
source $1

set -e # Error interrupt encountered

# check env
cd $One_YOLOv5_ROOT
ProjectPath=${One_YOLOv5_ROOT}/${Test_Ci_Project}/train-cls 
rm $ProjectPath -rf



echo "Test classify"
b=$ProjectPath/exp/weights/best  # best.of checkpoint

python classify/train.py  --imgsz 64 --batch 32 --model ${Weights}-cls.of --epochs 1  --project $ProjectPath      # train

python classify/val.py  --batch 32 --weights ${b}.of --device ${Device} --data  /home/fengwen/datasets/imagenette160                                             # val

python classify/predict.py --weights ${b}.of --device ${Device}   --source data/images/                                      # detect

python export.py --weights $b.of --img 64 --include onnx --device ${Device}                                         # export