#!/bin/bash

source $1

set -e # Error interrupt encountered

# check env
cd $One_YOLOv5_ROOT
ProjectPath=${One_YOLOv5_ROOT}/${Test_Ci_Project}/train-seg 
# rm $ProjectPath -rf


echo "Test segmentation"
m=${Model}-seg 
b=$P{rojectPath}/exp/weights/best  # best.of checkpoint
w=${Weights}-seg   # official weights
# python segment/train.py --project $ProjectPath  --imgsz 64 --batch 32 --weights $w.of  --epochs 1 --device $Device     # train

# # python segment/train.py --project $ProjectPath  --imgsz 64 --batch 32 --weights ' ' --cfg $Model-seg.yaml --epochs 1 --device $Device     # train

# python segment/val.py    --imgsz 64 --batch 32 --weights $w.of --device $Device                             # val

# python segment/predict.py   --imgsz 64 --weights $w.of --device $Device  # predict

python export.py  --weights $w.of --img 64 --include onnx --device $Device  # export

# export onnx Test 
