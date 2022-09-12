# 三 如何准备yolov5模型训练数据(以coco数据集和自定义数据集为例)

# Custom Training with YOLOv5

在本教程中，我们组装了一个数据集，并训练了一个自定义的YOLOv5模型来识别数据集中的对象。为此，我们将采取以下步骤：

* Gather a dataset of images and label our dataset
* Export our dataset to YOLOv5
* Train YOLOv5 to recognize the objects in our dataset
* Evaluate our YOLOv5 model's performance
* Run test inference to view our model at work



![](https://uploads-ssl.webflow.com/5f6bc60e665f54545a1e52a5/615627e5824c9c6195abfda9_computer-vision-cycle.png)


## coco数据集介绍
- 目录结构
- 一些细节
                                                                                                                                                                                                                                                                                       
## 参考文章
- https://blog.roboflow.com/train-yolov5-classification-custom-data/#prepare-a-custom-dataset-for-classification
- https://colab.research.google.com/github/roboflow-ai/yolov5-custom-training-tutorial/blob/main/yolov5-custom-training.ipynb#scrollTo=hrsaDfdVHzxt



# 训练自定义数据 📌
📚 本指南介绍如何使用YOLOv5训练您自己的自定义数据集 🚀.

# 开始之前

克隆此仓库，下载教程数据集， 和安装 [requirements](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) 依赖,包括 **Python>=3.8 and PyTorch>=1.7** 

```shell
$ git clone https://github.com/ultralytics/yolov5  # clone repo
$ cd yolov5
$ pip install -r requirements.txt  # install
```
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
使用工具比如 [CVAT](https://github.com/opencv/cvat) , [makesense.ai](https://www.makesense.ai/), [Labelbox](https://labelbox.com/) 去做标签在你自己的图片上，将标签导出为YOLO格式，带一个*.txt 的图像文件 （如果图像中没有对象，则不需要*.txt文件）。

*.txt文件规范如下所示:
- 每一行 一个对象。
- 每一行是 class x_center y_center width height 格式。
- 框坐标必须采用标准化xywh格式（从0到1）。如果框以像素为单位，则将x_center和width除以图像宽度，将y_centre和height除以图像高度。
- 类号为零索引的编号（从0开始计数）。

![imgs](https://user-images.githubusercontent.com/26833433/91506361-c7965000-e886-11ea-8291-c72b98c25eec.jpg)

与上述图像相对应的标签文件包含2个人（class 0）和 一个领带（class 27）：


![imgs](https://user-images.githubusercontent.com/26833433/112467037-d2568c00-8d66-11eb-8796-55402ac0d62f.png)

## 3.目录结构

组织你的train和val图片和标签 根据下面的示例。在本例中，我们假设 **/coco128**是位于 **/yolov5** 目录附近。YOLOv5通过将每个图像路径中的 **/images/** 的最后一个实例替换为 **/labels/** 来自动定位每个图像的标签。例如：
```Python
dataset/images/im0.jpg  # image
dataset/labels/im0.txt  # label
```
![imgs](https://user-images.githubusercontent.com/26833433/112467887-e18a0980-8d67-11eb-93af-6505620ff8aa.png)

4.选择模型

选择要开始训练的预训练模型。在这里，我们选择了[YOLOv5s](https://github.com/ultralytics/yolov5/blob/master/models/yolov5s.yaml)，这是可用的最小和最快的型号。有关所有模型的完整比较，具体请参阅[所以模型的比较表](https://github.com/ultralytics/yolov5#pretrained-checkpoints)。

![img](https://github.com/ultralytics/yolov5/releases/download/v1.0/model_comparison.png)

## 5. Train
在COCO128上训练YOLOv5s模型 通过指定数据集、批量大小、图像大小和预训练的 **--weights yolov5s.pt** (推荐),或随机初始化 **--weights  ' ' --cfg yolov5s.yaml** （不推荐）。预训练重量自动从[最新的YOLOv5版本下载](https://github.com/ultralytics/yolov5/releases)。

```python
# Train YOLOv5s on COCO128 for 5 epochs 
$ python train.py --img 640 --batch 16 --epochs 5 --data coco128.yaml --weights yolov5s.pt
```


所有训练结果都保存到 **runs/train/** 中，并带有递增的运行目录，即 **runs/train/exp2** 、**runs/train/exp3** 等。有关更多详细信息，请参阅我们的Google Colab笔记本的培训部分。[在Colab开放](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb)，[在Kaggle开放](https://colab.research.google.com/github/roboflow-ai/yolov5-custom-training-tutorial/blob/main/yolov5-custom-training.ipynb#scrollTo=ZbUn4_b9GCKO)

# 可视化(Visualize)
## Weights & Biases Logging (🚀 NEW)
Weights & Biases (W&B)现在与YOLOv5集成，用于实时可视化和训练运行的云记录。这允许更好的运行比较和内省，以及提高团队成员之间的可见性和协作。要启用W&B日志记录，请i**nstall wandb**，然后正常培训（第一次使用时将引导您进行设置）。
```Python
$ pip install wandb
```
在训练期间，您将在网页位置看到实时运行结果：https://wandb.ai，您可以使用 [W&B报告工具](https://wandb.ai/glenn-jocher/yolov5_tutorial/reports/YOLOv5-COCO128-Tutorial-Results--VmlldzozMDI5OTY) 创建结果的详细报告。

![img](https://user-images.githubusercontent.com/26833433/112469341-a8eb2f80-8d69-11eb-959a-dd85d3997bcf.jpg)

## 本地日志(Local Logging)

默认情况下，所有结果都记录为runs/train，并为每个新训练创建一个新的训练结果目录，如runs/train/exp2、runs/train/exp3等。查看训练和测试JPG以查看 mosaics, labels, predictions and augmentation 效果。
注意：Mosaic Dataloader 用于训练（如下所示），这是Ultralytics发表的新概念，首次出现在[YOLOv4](https://arxiv.org/abs/2004.10934)中。

**train_batch0.jpg** 显示 batch 为 0 的 (mosaics and labels):

![img](https://user-images.githubusercontent.com/26833433/83667642-90fcb200-a583-11ea-8fa3-338bbf7da194.jpeg)


test_batch0_labels.jpg 展示测试 batch 为 0 labels:

![img](https://user-images.githubusercontent.com/26833433/83667626-8c37fe00-a583-11ea-997b-0923fe59b29b.jpeg)

test_batch0_pred.jpg 展示测试 batch 为 0 predictions(预测):
![img](https://user-images.githubusercontent.com/26833433/83667635-90641b80-a583-11ea-8075-606316cebb9c.jpeg)


训练训损失和性能指标也记录到Tensorboard和自定义结果中**results.txt日志文件**，训练训完成后作为结果绘制 results.png如下。在这里，我们展示了在COCO128上训练的YOLOV5
- 从零开始训练 (蓝色)。
- 加载预训练权重 --weights yolov5s.pt (橙色)。

![img](https://user-images.githubusercontent.com/26833433/97808309-8182b180-1c66-11eb-8461-bffe1a79511d.png)

