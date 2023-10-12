python segment/train.py \
        --weights ../datasets/weights/yolov5s-seg.pt \
        --data data/coco128-seg.yaml \
        --project checkpoint \
        --name exp \
        --epochs 5 \
	--batch-size 8 \
        --img 640
