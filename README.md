

<center> 
<img src="https://user-images.githubusercontent.com/35585791/205295580-d1259bdd-14ff-4482-b741-e4bb49734dfa.png">
</center>

> 写在前面：本项目是基于 ultralytics 版 YOLOv5 源码改成 OneFlow 后端的结果，本工程的目的是做一个拥有更快训练速度的 YOLOv5 ，同时提供丰富的中文教程和源码细节解读，使得读者可以更加深入的了解 YOLOv5 。本 README 的其中一些部分也是直接用的 ultralytics 版 YOLOv5 README 的翻译，我们将相关链接替换为了 OneFlow 后端 YOLOv5 对应的链接。

# 最近新闻

- 🌟 v1.1.0 正式开源。详情请看：[Release Note](https://github.com/Oneflow-Inc/one-yolov5/releases/tag/v1.1.0)

为了说明使用 OneFlow 训练目标检测模型的可行性以及性能的优越性，最近我们将 [ultralytics 版 YOLOv5](https://github.com/ultralytics/yolov5) 通过 import oneflow as torch 的方式迁移为了OneFlow后端（我们将尽量跟随 ultralytics/yolov5 开源的新Feature，比如对分类，分割的支持）。并对 YOLOv5 中相关的教程进行了汉化，添加了一系列详细的代码解读，原理讲解以及部署教程，希望使得 YOLOv5 项目对用户更加透明化。另外我们也将在性能这个角度进行深入探索，我们在单卡上凭借对YOLOv5的性能分析以及几个简单的优化将GTX 3090 FP32 YOLOv5s的训练速度提升了近20%。对于需要迭代300个Epoch的COCO数据集来说相比 ultralytics/yolov5 我们缩短了11.35个小时的训练时间。详情请看[消费级显卡的春天，GTX 3090 YOLOv5s单卡完整训练COCO数据集缩短11.35个小时](https://start.oneflow.org/oneflow-yolo-doc/tutorials/00_chapter/optim_speed_version1.html)

- 🎉 [代码仓库地址](https://github.com/Oneflow-Inc/one-yolov5)
- 🎉 [文档网站地址](https://start.oneflow.org/oneflow-yolo-doc/index.html)
- 🎉 [OneFlow 安装方法](https://github.com/Oneflow-Inc/oneflow#install-oneflow) (注意⚠️： 目前OneFlow 需要安装 nightly 版本，等OneFlow后续新版本发布后可以安装稳定版。此外 OneFlow 目前仅对Linux平台有完整支持，请 Windows 和 Macos 用户注意)

不过即使你对 OneFlow 带来的性能提升不感兴趣，我们相信[OneYOLOv5文档网站](https://start.oneflow.org/oneflow-yolo-doc/index.html)中对 ultralytics/yolov5 相关教程的汉化以及源码剖析也会是从零开始深入学习 YOLOv5 一份不错的资料。欢迎在仓库给我们提出宝贵的意见。🌟🌟🌟

## <div align="center">文档</div>

请查看 [文档网站](https://start.oneflow.org/oneflow-yolo-doc/index.html) 获取关于训练，测试和推理的完整文档。

# 快速开始案例

注意⚠️:

- oneflow目前不支持windows平台

- --batch 必须是GPU数量的倍数。

- GPU 0 将比其他GPU占用略多的内存，因为它维护EMA并负责检查点等。


## <div align="center">快速开始案例</div>

<details open>
<summary>安装</summary>

在[**Python>=3.7.0**](https://www.python.org/) 的环境中克隆版本仓并安装 [requirements.txt](https://github.com/Oneflow-Inc/one-yolov5/blob/main/requirements.txt)，包括 [OneFlow nightly或者oneflow>=0.9.0](https://pytorch.org/get-started/locally/) 。


注意⚠️： 目前OneFlow 需要安装 nightly 版本，等OneFlow后续新版本发布后可以安装稳定版。

```bash
git clone https://github.com/Oneflow-Inc/one-yolov5  # 克隆
cd one-yolov5
pip install -r requirements.txt  # 安装
```

</details>


<details open>
<summary>推理</summary>

YOLOv5 的 OneFlow Hub 推理. [模型](https://github.com/Oneflow-Inc/one-yolov5/tree/main/models) 自动从最新YOLOv5 [版本](https://github.com/Oneflow-Inc/one-yolov5/releases)下载。

```python
import oneflow as flow

# 模型
model = flow.hub.load('Oneflow-Inc/one-yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

# 图像
img = 'https://raw.githubusercontent.com/Oneflow-Inc/one-yolov5/main/data/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# 推理
results = model(img)

# 结果
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

</details>

<details>
<summary>用 detect.py 进行推理</summary>

`detect.py` 在各种数据源上运行推理, 其会从最新的 YOLOv5 [版本](https://github.com/Oneflow-Inc/one-yolov5/releases) 中自动下载 [模型](https://github.com/Oneflow-Inc/one-yolov5/tree/main/models) 并将检测结果保存到 `runs/detect` 目录。

```bash
python detect.py --source 0  # 网络摄像头
                          img.jpg  # 图像
                          vid.mp4  # 视频
                          path/  # 文件夹
                          'path/*.jpg'  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP 流
```

</details>

<details>
<summary>训练</summary>

以下指令再现了 YOLOv5 [COCO](https://github.com/Oneflow-Inc/one-yolov5/blob/main/data/scripts/get_coco.sh)
数据集结果. [模型](https://github.com/Oneflow-Inc/one-yolov5/tree/main/models) 和 [数据集](https://github.com/Oneflow-Inc/one-yolov5/tree/main/data) 自动从最新的YOLOv5 [版本](https://github.com/Oneflow-Inc/one-yolov5/releases) 中下载。YOLOv5n/s/m/l/x的训练时间在V100 GPU上是 1/2/4/6/8天（多GPU倍速）. 尽可能使用最大的 `--batch-size`, 或通过 `--batch-size -1` 来实现 YOLOv5 [自动批处理](https://github.com/Oneflow-Inc/one-yolov5/blob/main/utils/autobatch.py#L21) 批量大小显示为 V100-16GB。

```bash
python train.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128
                                       yolov5s                                64
                                       yolov5m                                40
                                       yolov5l                                24
                                       yolov5x                                16
```

<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">

</details>

<details open>
<summary>教程和源码解读</summary>

- [0. one-yolov5特点解析](https://start.oneflow.org/oneflow-yolo-doc/tutorials/00_chapter/overview.html)
- [1. 消费级显卡的春天，GTX 3090 YOLOv5s单卡完整训练COCO数据集缩短11.35个小时](https://start.oneflow.org/oneflow-yolo-doc/tutorials/00_chapter/optim_speed_version1.html)
- [2. YOLOv5 网络结构解析](https://start.oneflow.org/oneflow-yolo-doc/tutorials/01_chapter/yolov5_network_structure_analysis.html)
- [3. 如何准备yolov5模型训练数据](https://start.oneflow.org/oneflow-yolo-doc/tutorials/02_chapter/how_to_prepare_yolov5_training_data.html)
- [4. 快速开始](https://start.oneflow.org/oneflow-yolo-doc/tutorials/03_chapter/quick_start.html)
- [5. 从OneFlow Hub 加载YOLOv5](https://start.oneflow.org/oneflow-yolo-doc/tutorials/03_chapter/loading_model_from_oneflowhub.html)
- [6. 测试时增强 (TTA)](https://start.oneflow.org/oneflow-yolo-doc/tutorials/03_chapter/TTA.html)
- [7. 模型融合 (Model Ensembling)](https://start.oneflow.org/oneflow-yolo-doc/tutorials/03_chapter/model_ensembling.html)
- [8. 数据增强](https://start.oneflow.org/oneflow-yolo-doc/tutorials/04_chapter/mosaic.html)
- [9. 矩形推理](https://start.oneflow.org/oneflow-yolo-doc/tutorials/05_chapter/rectangular_reasoning.html)
- [10. IOU深入解析](https://start.oneflow.org/oneflow-yolo-doc/tutorials/05_chapter/iou_in-depth_analysis.html)
- [11. 模型精确度评估](https://start.oneflow.org/oneflow-yolo-doc/tutorials/05_chapter/map_analysis.html)
- [12. 计算mAP用到的numpy函数](https://start.oneflow.org/oneflow-yolo-doc/tutorials/05_chapter/Introduction_to_functions_used_in_metrics.html)
- [13. ONNX模型导出](https://start.oneflow.org/oneflow-yolo-doc/tutorials/06_chapter/export_onnx_tflite_tensorrt.html)
- [14. train.py源码解读](https://start.oneflow.org/oneflow-yolo-doc/source_code_interpretation/train_py.html) 

持续新增中...

</details>



## <div align="center">为什么选择 one-yolov5</div>

[消费级显卡的春天，GTX 3090 YOLOv5s单卡完整训练COCO数据集缩短11.35个小时](https://start.oneflow.org/oneflow-yolo-doc/tutorials/00_chapter/optim_speed_version1.html)

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040763-93c22a27-347c-4e3c-847a-8094621d3f4e.png"></p>
<details>
  <summary>YOLOv5-P5 640 图像 (点击扩展)</summary>

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040757-ce0934a3-06a6-43dc-a979-2edbbd69ea0e.png"></p>
</details>
<details>
  <summary>图片注释 (点击扩展)</summary>

- **COCO AP val** 表示 mAP@0.5:0.95 在5000张图像的[COCO val2017](http://cocodataset.org)数据集上，在256到1536的不同推理大小上测量的指标。
- **GPU Speed** 衡量的是在 [COCO val2017](http://cocodataset.org) 数据集上使用 [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) V100实例在批量大小为32时每张图像的平均推理时间。
- **EfficientDet** 数据来自 [google/automl](https://github.com/google/automl) ，批量大小设置为 8。
- 复现 mAP 方法: `python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n6 yolov5s6 yolov5m6 yolov5l6 yolov5x6`

</details>

### 预训练检查点

| 模型                                                                                                 | 规模<br><sup>(像素) | mAP<sup>验证<br>0.5:0.95 | mAP<sup>验证<br>0.5 | 速度<br><sup>CPU b1<br>(ms) | 速度<br><sup>V100 b1<br>(ms) | 速度<br><sup>V100 b32<br>(ms) | 参数<br><sup>(M) | 浮点运算<br><sup>@640 (B) |
|------------------------------------------------------------------------------------------------------|---------------------|--------------------------|---------------------|-----------------------------|------------------------------|-------------------------------|------------------|---------------------------|
| [YOLOv5n](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5n.zip)                   | 640                 | 28.0                     | 45.7                | **45**                      | **6.3**                      | **0.6**                       | **1.9**          | **4.5**                   |
| [YOLOv5s](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5s.zip)                   | 640                 | 37.4                     | 56.8                | 98                          | 6.4                          | 0.9                           | 7.2              | 16.5                      |
| [YOLOv5m](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5m.zip)                   | 640                 | 45.4                     | 64.1                | 224                         | 8.2                          | 1.7                           | 21.2             | 49.0                      |
| [YOLOv5l](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5l.zip)                   | 640                 | 49.0                     | 67.3                | 430                         | 10.1                         | 2.7                           | 46.5             | 109.1                     |
| [YOLOv5x](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5x.zip)                   | 640                 | 50.7                     | 68.9                | 766                         | 12.1                         | 4.8                           | 86.7             | 205.7                     |
|                                                                                                      |                     |                          |                     |                             |                              |                               |                  |                           |
| [YOLOv5n6](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5n6.zip)                 | 1280                | 36.0                     | 54.4                | 153                         | 8.1                          | 2.1                           | 3.2              | 4.6                       |
| [YOLOv5s6](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5s6.zip)                 | 1280                | 44.8                     | 63.7                | 385                         | 8.2                          | 3.6                           | 12.6             | 16.8                      |
| [YOLOv5m6](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5m6.zip)                 | 1280                | 51.3                     | 69.3                | 887                         | 11.1                         | 6.8                           | 35.7             | 50.0                      |
| [YOLOv5l6](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5l6.zip)                 | 1280                | 53.7                     | 71.3                | 1784                        | 15.8                         | 10.5                          | 76.8             | 111.4                     |
| [YOLOv5x6](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5x6.zip)<br>+ [TTA][TTA] | 1280<br>1536        | 55.0<br>**55.8**         | 72.7<br>**72.7**    | 3136<br>-                   | 26.2<br>-                    | 19.4<br>-                     | 140.7<br>-       | 209.8<br>-                |

<details>
  <summary>表格注释 (点击扩展)</summary>

- 所有检查点都以默认设置训练到300个时期. Nano和Small模型用 [hyp.scratch-low.yaml](https://github.com/Oneflow-Inc/one-yolov5/blob/master/data/hyps/hyp.scratch-low.yaml) hyps, 其他模型使用 [hyp.scratch-high.yaml](https://github.com/Oneflow-Inc/one-yolov5/blob/master/data/hyps/hyp.scratch-high.yaml).
- **mAP<sup>val</sup>** 值是 [COCO val2017](http://cocodataset.org) 数据集上的单模型单尺度的值。
<br>复现方法: `python val.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`
- 使用 [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) 实例对COCO val图像的平均速度。不包括NMS时间（~1 ms/img)
<br>复现方法: `python val.py --data coco.yaml --img 640 --task speed --batch 1`
- **TTA** [测试时数据增强](https://github.com/ultralytics/yolov5/issues/303) 包括反射和比例增强. # 文档网站还没有,稍后更新。
<br>复现方法: `python val.py --data coco.yaml --img 1536 --iou 0.7 --augment`

</details>
