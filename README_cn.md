# One-YOLO: OneFlow Based YOLO Seris Models

One-YOLO 算法库是基于 OneFlow 深度学习框架和 Ultralytics/yolov5 仓库实现的一系列 YOLO 检测算法组合工具箱。在保持和 Ultralytics/yolov5 仓库**代码结构以及使用方式完全相同**的基础上，我们将从以下几个角度对原始的 Ultralytics/yolov5 进行扩展：

## 横向扩展（学术研究，实验更加方便）

- 支持更多的 YOLO 系列检测模型，如 YOLOv3 ，YOLOV4 ，YOLOV5，YOLOv6，YOLOV7，YOLOX ...
- 支持更多的 BackBone，Neck，检测头，注意力机制，Loss，Iou Loss，NMS 策略，数据增强方式 ... 

具体支持情况在下表动态更新

| 支持类型 | feature 名称 |feature 出处| 是否已经支持 | 功能是否验证|Feature讲解的文章或者视频|
| --     | --         | --         | --         |     --       |    --     |
| 内置模型 | YOLOV3     | ✅         |            |              |           |
| 内置模型 | YOLOV3-Tiny|            |            |              |           |
| 内置模型 | YOLOV4     |            |            |              |           |
| 内置模型 | YOLOV5     |            |            |              |           |   
| 内置模型 | YOLOV6     |            |            |              |           |
| 内置模型 | YOLOV7     |            |            |              |           |
| BackBone| CSPDarkNet系列 |        |             |             |           |
| BackBone| MobileNet系列  |        |             |             |           |
| BackBone| EfficientNet系列|        |             |             |           |

...

## 纵向扩展（工程部署，编译优化更加方便）

- 对所有的内置模型和不同的 BackBone 都支持导出到 ONNX，TensorRT，OpenVINO 等推理框加速目标平台的推理
- 和深度学习编译器如 TVM 进行交互，使用 TVM 尝试部署一些模型，探索深度学习编译器的落地可行性
- 接入 openppl-public ppq ，解决离线 INT8 量化检测模型并部署到目标平台的问题

具体支持情况在下表动态更新：

| 模型 | 导出 onnx | 导出 openvino | 导出 TensorRT | ppq 实践（代码链接） | TVM 实践（代码链接） |
| --     | --         | --         | --         |     --       |    --     |
| YOLOV3 |            |            |            |              |           |
| YOLOV3-Tiny|        |            |            |              |           |
| YOLOV4 |            |            |            |              |           |
| YOLOV5 |            |            |            |              |           |
| YOLOV6 |            |            |            |              |           |
| YOLOV7 |            |            |            |              |           |

...


## 使用

### 安装

### 训练

### 推理

## BenchMark

## 教程

## 将来计划

## Acknowledgements

<details><summary> <b>展开</b> </summary>

[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)  
[https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)  
[https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)  
[https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)  
[https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)   
[https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)  
[https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)  
[https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)   
[https://github.com/xmu-xiaoma666/External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch)  
[https://gitee.com/SearchSource/yolov5_yolox](https://gitee.com/SearchSource/yolov5_yolox)  
[https://github.com/Krasjet-Yu/YOLO-FaceV2](https://github.com/Krasjet-Yu/YOLO-FaceV2)  
[https://github.com/positive666/yolov5_research/](https://github.com/positive666/yolov5_research)  
[https://github.com/ppogg/YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite)  
[https://github.com/Gumpest/YOLOv5-Multibackbone-Compression](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression)  
[https://github.com/cv516Buaa/tph-yolov5](https://github.com/cv516Buaa/tph-yolov5)
Paper:[https://arxiv.org/abs/2208.02019](https://arxiv.org/abs/2208.02019)  

</details>
