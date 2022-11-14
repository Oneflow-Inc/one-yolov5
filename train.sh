python  -m oneflow.distributed.launch \
--nproc_per_node 1 \
train.py \
--batch 2 \
--cfg models/hub/yolov5s6.yaml \
 --weights '' \
 --data data/coco.yaml \
 --img 640 \
 --epochs 1  \
 --hyp  data/hyps/hyp.scratch-high.yaml \
#  --sync-bn
