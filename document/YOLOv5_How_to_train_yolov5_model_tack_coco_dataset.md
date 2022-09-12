# 三 如何准备yolov5模型训练数据(以coco数据集和自定义数据集为例)
## 前言
本章主要以介绍 训练在自定义数据集，到如何进行数据集制作能获得更好的训练结果进行概述。

# 训练在自定义数据集
## 1.创建dataset.yaml
COCO128是官方给的一个小的数据集 由[COCO](https://cocodataset.org/#home)数据集前128张图片组成。
这128幅图像用于训练和验证，以验证我们的训练是否能够过正常进行。[data/coco128.yaml](https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml) 。
如下所示，数据集配置文件定义了选项 

1) 用于自动下载的可选下载命令/URL 。 
2) 指向训练图像目录的路径（或指向包含训练图像列表的*.txt文件的路径
3) 验证图像的路径相同
4) 类数
5) 类名列表；
```coco128.yaml
# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]


train: ../coco128/images/train2017/
val: ../coco128/images/train2017/

# number of classes
nc: 80 # 类数

# class names 类名列表
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
        'teddy bear', 'hair drier', 'toothbrush']

   ```
## 2.创建 Labels
使用工具例如 [CVAT](https://github.com/opencv/cvat) , [makesense.ai](https://www.makesense.ai/), [Labelbox](https://labelbox.com/) ，labelimg(在本章如何制作数据集中介绍labelimg工具使用) 等，去做标签在你自己的图片上，将标签导出为YOLO格式，带一个*.txt 的图像文件 （如果图像中没有对象，则不需要*.txt文件）。

*.txt文件规范如下所示:
- 每一行 一个对象。
- 每一行是 class x_center y_center width height 格式。
- 框坐标必须采用标准化xywh格式（从0到1）。如果框以像素为单位，则将x_center和width除以图像宽度，将y_centre和height除以图像高度。
- 类号为零索引的编号（从0开始计数）。


<p align="center">
  <img src="https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f5106cf39a3f44fe997d488b67d5da83~tplv-k3u1fbpfcp-zoom-1.image">
</p>
与上述图像相对应的标签文件包含2个人（class 0）和 一个领带（class 27）：


![imgs](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/270bc6d3bb0b406fa12f4a83a763819b~tplv-k3u1fbpfcp-zoom-1.image)

## 3.目录结构

组织你的train和val图片和标签 根据下面的示例。在本例中，我们假设 **/coco128**是位于 **/yolov5** 目录附近。YOLOv5通过将每个图像路径中的 **/images/** 的最后一个实例替换为 **/labels/** 来自动定位每个图像的标签。例如：
```Python
dataset/images/im0.jpg  # image
dataset/labels/im0.txt  # label
```
![imgs](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/cf60a753e49b4982a212f0e28b727df9~tplv-k3u1fbpfcp-zoom-1.image)


# 制作数据集

## 数据集标注工具
这里主要介绍 labelimg: 是一种矩形标注工具，常用于目标识别和目标检测,可直接生成yolo读取的txt标签格式，但其只能进行矩形框标注。(当然也可以选用其它的工具进行标注并且网上都有大量关于标注工具的教程。)

首先labelimg的安装十分简单，直接使用cmd中的pip进行安装，在cmd中输入命令行：
```python3
pip install labelimg
```
安装后直接输入命令：

labelimg

即可打开运行：


<p align="center">
  <img src="https://github.com/Oneflow-Inc/one-yolov5/blob/how-to-prepare-yolov5-model-training-data/data/images/labelimg.png">
</p>




## 一个好的数据集
- 每个类的图像。 >= 1500 张图片。
- 每个类的实例。≥ 建议每个类10000个实例（标记对象）
- 图片形象多样。必须代表已部署的环境。对于现实世界的使用案例，我们推荐来自一天中不同时间、不同季节、不同天气、不同照明、不同角度、不同来源（在线采集、本地采集、不同摄像机）等的图像。
- 标签一致性。必须标记所有图像中所有类的所有实例。部分标记将不起作用。
- 标签准确性。
- 标签必须紧密地包围每个对象。对象与其边界框之间不应存在任何空间。任何对象都不应缺少标签。
- 标签验证。查看train_batch*.jpg 在 训练开始验证标签是否正确，即参见 mosaic。
- 背景图像。背景图像是没有添加到数据集以减少 False Positives（FP）的对象的图像。我们建议使用大约0-10%的背景图像来帮助减少FPs（COCO有1000个背景图像供参考，占总数的1%）。背景图像不需要标签。




<p align="center">
  <a href= "https://arxiv.org/abs/1405.0312">
  <img src="https://user-images.githubusercontent.com/26833433/109398377-82b0ac00-78f1-11eb-9c76-cc7820669d0d.png">
  </a>
</p>


## 参考文章
https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results
https://docs.ultralytics.com/tutorials/train-custom-datasets/#weights-biases-logging-new
