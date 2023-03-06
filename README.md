

<center> 
<img src="https://user-images.githubusercontent.com/35585791/223076182-abdca39b-3084-472d-a377-7bbaa640b6f0.png">
</center>

> 写在前面：本项目是基于 ultralytics 版 YOLOv5 源码改成 OneFlow 后端的结果，本工程的目的是做一个拥有更快训练速度的 YOLOv5 ，同时提供丰富的中文教程和源码细节解读，使得读者可以更加深入的了解 YOLOv5 。本 README 的其中一些部分也是直接用的 ultralytics 版 YOLOv5 README 的翻译，我们将相关链接替换为了 OneFlow 后端 YOLOv5 对应的链接。

### <div align="center">最近新闻</div>

- 🌟 v1.2.0 正式开源。v1.2.0同步了ultralytics yolov5的v7.0版本，同时支持分类，目标检测，实例分割等任务 详情请看：[Release Note](https://github.com/Oneflow-Inc/one-yolov5/releases/tag/v1.2.0)
<table border="1px" cellpadding="10px">
        <tr>
            <td>
                <img src="https://user-images.githubusercontent.com/109639975/220929631-9baf1d12-8cfc-4e9f-985e-372302b672dc.jpg" height="280px"  width="575px"  >
            </td>
            <td>
                <img src="https://user-images.githubusercontent.com/109639975/220928826-84ed25bc-a72e-46ab-8b9c-c3a2b57ded18.jpg" height="280"  width="575px" >
            </td>
        </tr>
        <tr>
            <td>
                <img src="https://user-images.githubusercontent.com/109639975/220929320-9f4cf581-43b9-4609-8b51-346c84ac0d62.jpg" height="280"  width="575px" >
            </td>
            <td>
                <img src="https://user-images.githubusercontent.com/109639975/220930143-aa022378-4b6f-4ffc-81bf-3e6032d4862c.jpg" height="280"  width="575px" >
            </td>
        </tr>
        <tr  >
            <td >
                原图 
            </td>
            <td  >
               目标检测: 目标检测是指从图像中检测出多个物体并标记它们的位置和类别。目标检测任务需要给出物体的类别和位置信息，通常使用边界框（bounding box）来表示。目标检测可以应用于自动驾驶、视频监控、人脸识别等领域。
            </td>
        </tr>
        <tr  >
            <td >
               图像分类:  图像分类是指给定一张图像，通过计算机视觉技术来判断它属于哪一类别。
图像分类是一种有监督学习任务，需要通过训练样本和标签来建立分类模型。在图像分类中，算法需要提取图像的特征，然后将其分类为预定义的类别之一。例如，图像分类可以用于识别手写数字、识别动物、区分汽车和自行车等。
            </td>
            <td >
            实例分割: 实例分割是指从图像中检测出多个物体并标记它们的位置和类别，同时对每个物体进行像素级的分割。
实例分割要求更为精细的信息，因为它需要将物体的每个像素都分配给对应的物体。 
实例分割可以应用于医学影像分析、自动驾驶、虚拟现实等领域。
            </td>
        </tr>
    </table>

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

在[**Python>=3.7.0**](https://www.python.org/) 的环境中克隆版本仓并安装 [requirements.txt](https://github.com/Oneflow-Inc/one-yolov5/blob/main/requirements.txt)，包括 [OneFlow nightly 或者 oneflow>=0.9.0](https://docs.oneflow.org/master/index.html) 。


```bash
git clone https://github.com/Oneflow-Inc/one-yolov5  # 克隆
cd one-yolov5
pip install -r requirements.txt  # 安装
```

</details>


### Train 
YOLOv5实例分割模型支持使用 `--data coco128-seg.yaml`  参数自动下载 `COCO128-seg` 测试数据集(*测试数据集表示能测试项目正常运行的小数据集*)， 以及使用 `bash data/scripts/get_coco.sh --train --val --segments`  或者使用  `python train.py --data coco.yaml`  下载 `COCO-segments` 数据集

```shell
# Single-GPU
python segment/train.py --model yolov5s-seg.of --data coco128-seg.yaml --epochs 5 --img 640

# Multi-GPU DDP
python -m oneflow.distributed.launch --nproc_per_node  4  segment/train.py --model yolov5s-seg.of --data coco128-seg.yaml --epochs 5 --img 640 --device 0,1,2,3
```

注意 :
- {`.of`: 代表OneFlow预训练权重 , `.pt`: 代表 PyTorch 预训练权重 }
- `--model yolov5s-seg.of`  表示使用OneFlow预训练权重 , 也是支持使用 PyTorch 预训练权重 如 `--model yolov5s-seg.pt`
- 模型权重将自动从 github 下载(*建议如果没有设置代理，可以提前将模型下载到电脑本地 使用 `--model 本地路径/yolov5s-seg.of`*)

### val 

在ImageNet-1k数据集上验证YOLOv5m-seg 模型的精度

```shell 
bash data/scripts/get_coco.sh --val --segments  # download COCO val segments split (780MB, 5000 images)
python segment/val.py --weights yolov5s-seg.of --data coco.yaml --img 640  # validate
```

### Predict 

使用预训练模型(YOLOv5m-seg) 预测图片

```shell
python segment/predict.py --weights yolov5m-seg.of --data data/images/
```

![image](https://user-images.githubusercontent.com/118866310/223043320-ba3599d9-a3a4-4590-af98-65da1e3f228c.png)

### Export

将 `yolov5s-seg` 模型导出为 ONNX 格式 示例
```shell
python export.py --weights yolov5s-seg.of --include onnx  --img 640 --device 0
```




## <div align="center">为什么选择 one-yolov5</div>

[消费级显卡的春天，GTX 3090 YOLOv5s单卡完整训练COCO数据集缩短11.35个小时](https://start.oneflow.org/oneflow-yolo-doc/tutorials/00_chapter/optim_speed_version1.html)


### CheckPoints

| 模型                                                                                                        | ONNX版本模型                                                                                      | 规模<br><sup>(像素) | mAP<sup>验证<br>0.5:0.95 | mAP<sup>验证<br>0.5 | 速度<br><sup>CPU b1<br>(ms) | 速度<br><sup>V100 b1<br>(ms) | 速度<br><sup>V100 b32<br>(ms) | 参数<br><sup>(M) | 浮点运算<br><sup>@640 (B) |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------- | ------------------------ | ------------------- | --------------------------- | ---------------------------- | ----------------------------- | ---------------- | ------------------------- |
| [YOLOv5n](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5n.zip)                   | [YOLOv5n.onnx](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.1.0/yolov5n.onnx)   | 640                 | 28.0                     | 45.7                | **45**                      | **6.3**                      | **0.6**                       | **1.9**          | **4.5**                   |
| [YOLOv5s](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5s.zip)                   | [YOLOv5s.onnx](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.1.0/yolov5s.onnx)   | 640                 | 37.4                     | 56.8                | 98                          | 6.4                          | 0.9                           | 7.2              | 16.5                      |
| [YOLOv5m](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5m.zip)                   | [YOLOv5m.onnx](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.1.0/yolov5m.onnx)   | 640                 | 45.4                     | 64.1                | 224                         | 8.2                          | 1.7                           | 21.2             | 49.0                      |
| [YOLOv5l](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5l.zip)                   | [YOLOv5l.onnx](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.1.0/yolov5l.onnx)   | 640                 | 49.0                     | 67.3                | 430                         | 10.1                         | 2.7                           | 46.5             | 109.1                     |
| [YOLOv5x](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5x.zip)                   | [YOLOv5x.onnx](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.1.0/yolov5x.onnx)   | 640                 | 50.7                     | 68.9                | 766                         | 12.1                         | 4.8                           | 86.7             | 205.7                     |
|                                                                                                             |                                                                                                   |                     |                          |                     |                             |                              |                               |                  |                           |
| [YOLOv5n6](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5n6.zip)                 | [YOLOv5n6.onnx](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.1.0/yolov5n6.onnx) | 1280                | 36.0                     | 54.4                | 153                         | 8.1                          | 2.1                           | 3.2              | 4.6                       |
| [YOLOv5s6](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5s6.zip)                 | [YOLOv5s6.onnx](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.1.0/yolov5s6.onnx) | 1280                | 44.8                     | 63.7                | 385                         | 8.2                          | 3.6                           | 12.6             | 16.8                      |
| [YOLOv5m6](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5m6.zip)                 | [YOLOv5m6.onnx](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.1.0/yolov5m6.onnx) | 1280                | 51.3                     | 69.3                | 887                         | 11.1                         | 6.8                           | 35.7             | 50.0                      |
| [YOLOv5l6](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5l6.zip)                 | [YOLOv5l6.onnx](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.1.0/yolov5l6.onnx) | 1280                | 53.7                     | 71.3                | 1784                        | 15.8                         | 10.5                          | 76.8             | 111.4                     |
| [YOLOv5x6](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.0.0/yolov5x6.zip)<br>+ [TTA][TTA] | [YOLOv5x6.onnx](https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.1.0/yolov5x6.onnx) | 1280<br>1536        | 55.0<br>**55.8**         | 72.7<br>**72.7**    | 3136<br>-                   | 26.2<br>-                    | 19.4<br>-                     | 140.7<br>-       | 209.8<br>-                |


<details>
  <summary>表格注释 (点击扩展)</summary>

- 所有检查点都以默认设置训练到300个时期. Nano和Small模型用 [hyp.scratch-low.yaml](https://github.com/Oneflow-Inc/one-yolov5/blob/master/data/hyps/hyp.scratch-low.yaml) hyps, 其他模型使用 [hyp.scratch-high.yaml](https://github.com/Oneflow-Inc/one-yolov5/blob/master/data/hyps/hyp.scratch-high.yaml).
- **mAP<sup>val</sup>** 值是 [COCO val2017](http://cocodataset.org) 数据集上的单模型单尺度的值。
<br>复现方法: `python val.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`
- 使用 [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) 实例对COCO val图像的平均速度。不包括NMS时间（~1 ms/img)
<br>复现方法: `python val.py --data coco.yaml --img 640 --task speed --batch 1`
- **TTA** [测试时数据增强](https://github.com/ultralytics/yolov5/issues/303) 包括反射和比例增强.
<br>复现方法: `python val.py --data coco.yaml --img 1536 --iou 0.7 --augment`

</details>
