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

# echo "Test segmentation"
# m=${Weights}-seg  # official weights
# b=${Test_Ci_Project}/train_seg/exp/weights/best  # best.of checkpoint

# python segment/train.py --imgsz 64 --batch 32 --weights $m.of  --epochs 1 --device $Device  --project $Test_Ci_Project/train-seg   # train
# python segment/train.py --imgsz 64 --batch 32 --weights '' --cfg $Model-seg.yaml --epochs 1 --device $Device   --project $Test_Ci_Project/train-seg  # train

#     python segment/val.py --project $Test_Ci_Project   --imgsz 64 --batch 32 --weights $w.of --device $d  # val
#     python segment/predict.py --project $Test_Ci_Project  --imgsz 64 --weights $w.of --device $d  # predict
#     python export.py --project $Test_Ci_Project  --weights $w.of --img 64 --include onnx --device $d  # export



# python val.py --project $Test_Ci_Project --imgsz 64 --batch 32 --weights ${Weights}.of --device ${Device} # val
# python detect.py --project $Test_Ci_Project --imgsz 64 --weights ${Weights}.of --device ${Device}         # detect
# python export.py --weights $Weights.of --img 64 --include onnx --device ${Device} # export

# echo "Test predictions"
# python export.py --weights ${Weights}-cls.of --include onnx --img 224 --device ${Device}
# python detect.py --weights ${Weights}.onnx --img 320 --device ${Device}
# python segment/predict.py --weights ${Weights}-seg.onnx --img 320 --device ${Device}
# python classify/predict.py --weights ${Weights}-cls.onnx --img 224 --device ${Device}

# echo "Test detection"
# b=runs/train/exp/weights/best  # best.of checkpoint
# python train.py --imgsz 64 --batch 32 --weights $m.of --cfg $m.yaml --epochs 1  # train


# python - <<EOF
# import oneflow as torch
# im = torch.zeros([1, 3, 64, 64])
# for path in '$m', '$b':
#     model = torch.hub.load('.', 'custom', path=path, source='local')
#     print(model('data/images/bus.jpg'))
#     model(im)  # warmup, build grids for trace
# EOF

# echo "Test segmentation"
# m=${Weights}-seg  # official weights
# b=${Test_Ci_Project}/train-seg/exp/weights/best  # best.of checkpoint

# python segment/train.py --imgsz 64 --batch 32 --weights $m.of --cfg $m.yaml --epochs 1 --device $Device  # train
# python segment/train.py --imgsz 64 --batch 32 --weights '' --cfg $m.yaml --epochs 1 --device $Device  # train

# for w in $m $b; do  # weights
#     python segment/val.py --project $Test_Ci_Project   --imgsz 64 --batch 32 --weights $w.of --device $d  # val
#     python segment/predict.py --project $Test_Ci_Project  --imgsz 64 --weights $w.of --device $d  # predict
#     python export.py --project $Test_Ci_Project  --weights $w.of --img 64 --include onnx --device $d  # export
# done

# echo "Test classification"
# m=${Weights}-cls.of  # official weights
# b=${Test_Ci_Project}/train-cls/exp/weights/best.of  # best.of checkpoint

# python classify/train.py --project $Test_Ci_Project --imgsz 32 --model $m --data mnist160 --epochs 1  --device $Device # train
# python classify/val.py --project $Test_Ci_Project --imgsz 32 --weights $b --data ../datasets/mnist160 --device $Device # val
# python classify/predict.py --project $Test_Ci_Project --imgsz 32 --weights $b --source ../datasets/mnist160/test/7/60.png --device $Device # predict
# python classify/predict.py --project $Test_Ci_Project --imgsz 32 --weights $m --source data/images/bus.jpg --device $Device # predict

# python export.py --weights $b --img 64 --include onnx --device $Device # export

# python - <<EOF
# import oneflow as torch
# for path in '$m', '$b':
#     model = torch.hub.load('.', 'custom', path=path, source='local')