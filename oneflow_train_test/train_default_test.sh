#!/bin/bash

source $1

set -e # Error interrupt encountered

# check env
cd $One_YOLOv5_ROOT
ProjectPath=${One_YOLOv5_ROOT}/${Test_Ci_Project}/train-default 
rm $ProjectPath -rf



echo "Test default"
b=$ProjectPath/exp/weights/best  # best.of checkpoint
python train.py --imgsz 64 --batch 32 --weights ${Weights}.of --epochs 1  --project $ProjectPath               # train

# python train.py --imgsz 64 --batch 32 --weights ' ' --cfg ${Model}.yaml --epochs 1 --project $ProjectPath      # train

python val.py  --batch 32 --weights ${b}.of --device ${Device}                                                 # val

python detect.py --weights ${b}.of --device ${Device}                                                          # detect

python export.py --weights $b.of --img 64 --include onnx --device ${Device}                                    # export