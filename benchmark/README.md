## One-YOLOV5 BenchMark

### 单卡测试结果
- 以下为GTX 3080ti(12GB) 的yolov5测试结果（oneflow后端 vs PyTorch后端）
- 以下测试结果的数据配置均为coco.yaml，模型配置也完全一样，并记录训练完coco数据集1个epoch需要的时间。
- 由于oneflow eager目前amp的支持还不完善，所以我们提供的结果均为fp32模式下进行训练的性能结果。
- PyTorch版本 yolov5 code base链接：https://github.com/ultralytics/yolov5
- OneFlow版本 yolov5 code base链接：https://github.com/Oneflow-Inc/one-yolov5
- 测试的命令为：`python train.py --batch 16 --cfg models/yolov5n.yaml --weights '' --data coco.yaml --img 640 --device 0`

| 后端    | 网络    | batch-size | 训练完1个epoch的耗时（分钟） |
| ------- | ------- | ---------- | ---------------------------- |
| PyTorch | yolov5n | 16         | 9.54                         |
| OneFlow | yolov5n | 16         | 9.48                         |
| PyTorch | yolov5s | 16         | 15.12                        |
| OneFlow | yolov5s | 16         | 14.52                        |
| PyTorch | yolov5m | 8          | 34.2                         |
| OneFlow | yolov5m | 8          | 31.26                        |

单卡模式下对于yolov5n网络，以fp32模式在coco上训练一个epoch，oneflow相比于pytorch加速比为 。
单卡模式下对于yolov5s网络，以fp32模式在coco上训练一个epoch，oneflow相比于pytorch加速比为 。



### 多卡测试结果


| 后端    | 网络    | GPU数 | batch-size | 训练完1个epoch的耗时（分钟） |
| ------- | ------- | ----- | ---------- | ---------------------------- |
| PyTorch | yolov5n | 2     | 16         | 8.58                         |
| OneFlow | yolov5n | 2     | 16         | 8.16                         |
| PyTorch | yolov5s | 2     | 16         | 10.68                        |
| OneFlow | yolov5s | 2     | 16         | 10.2                         |
| PyTorch | yolov5m | 2     | 16         | 19.32                        |
| OneFlow | yolov5m | 2     | 16         | 17.58                        |
