本文翻译自: https://arxiv.org/pdf/1612.08242.pdf


$YOLO9000$ :
Better, Faster, Stronger

Joseph Redmon∗†, Ali Farhadi∗†

University of Washington∗, Allen Institute for AI†

http://pjreddie.com/yolo9000/

## 摘要
&emsp;我们推出的 $YOLO9000$ 是一款先进的实时目标检测系统，可检测9000多种目标类别。首先，我们提出对  $YOLO$  检测方法的各种改进，这些改进有独创的，也有的是来源于以前的研究。改进后的模型  $YOLOv2$  在 $PASCAL  VOC$ 和  $COCO$ 等标准检测任务中处于技术领先地位。通过使用一种新颖的多尺度训练方法，同样的  $YOLOv2$  模型可以以不同的尺寸运行，在速度和准确性之间提供了一个简单的折衷。在 $67 FPS$ 时，  $YOLOv2$  在 $VOC \ 2007$ 上获得了76.8 $mAP$ 。在 $40FPS$ 时，  $YOLOv2$  获得了78.6 $mAP$，超越了采用 $ResNet$ 和 $SSD的Faster \ R-CNN$ 等先进的方法，同时运行速度仍然更快。最后我们提出一种联合训练目标检测和分类的方法。使用这种方法，我们在 $COCO$ 检测数据集和 $ImageNet$ 分类数据集上同时训练 $YOLO9000$ 。我们的联合训练使 $YOLO9000$ 能够预测未标注检测数据的目标类别，并且我们在 $ImageNet$ 检测任务上验证了我们的方法。  $YOLO9000$ 在 $ImageNet$ 检测验证集上获得19.7  $mAP$  ，尽管200个类中只有44个具有检测数据。在 $COCO$ 上没有的156种类上， $YOLO9000$ 得到 16.0  $mAP$  ，但是 $YOLO$ 可以检测超过200个种类;它预测超过9000 多种不同的目标类别，而且它仍然是实时运行的。

<p>

<!-- <img src = "8_yolo_v1-v7/02_yolo_imgs/图片01.png"> -->
<img src = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/one-yolo/document/02_yolo_imgs/%E5%9B%BE%E7%89%8701.png">

图1: $YOLO9000$ 。 $YOLO9000$ 可以实时检测各种各样的目标类别。

</p>

## 1.介绍

&emsp;通用的目标检测应该快速，准确，并且能够识别各种各样的目标。自从引入神经网络，检测框架变得越来越快速和准确。但是，大多数检测方法仅限于检测一小部分目标。



&emsp;与分类和标记等其他任务的数据集相比，目前目标检测数据集是有限的。最常见的检测数据集包含成千上万到数十万张具有成百上千个标签的图像[3][10][2]。而分类数据集有数以百万计的图像，数十或数百万个类别[20][2]。



&emsp;我们希望检测的类别能够扩展到目标分类的级别。但是，标注检测图像要比标注分类或贴标签要昂贵得多（*标签通常是用户免费提供*) 。因此，我们不太可能在近期内看到与分类数据集相同规模的检测数据集。



&emsp;我们提出了一种新方法——通过利用我们已有的大量分类数据来扩大当前检测系统的范围。 我们的方法使用目标分类的分层视图，使得我们可以将不同的数据集组合在一起。

&emsp;我们还提出了一种联合训练算法，它允许我们在检测和分类数据上训练目标检测器。 我们的方法利用标记检测图像来学习精确定位目标，同时使用分类图像来增加词汇量和鲁棒性。

&emsp;我们使用这种方法训练 $YOLO9000$ b一种可以检测超过9000种不同的目标类别的实时目标检测器。 首先，我们改进YOLO基础检测系统，生成最先进的实时检测器  $YOLOv2$  。 然后，采用我们的数据集组合方法和联合训练算法，使用来自 $ImageNet$ 的9000多个类以及 $COCO$ 的检测数据来训练模型。

我们所有代码和预训练模型都可在线获得：http://pjreddie.com/yolo9000/。

## 2.更好

&emsp;与最先进的检测系统相比， $YOLO$ 存在各种缺点。 $YOLO$ 与 $Fast \ R-CNN$ 的误差比较分析表明， $YOLO$ 产生了大量的定位错误。此外，与生成候选区域方法相比，$YOLO$ 召回率(*recall*)相对较低。 因此，我们主要关注改善召回率和定位，同时保持分类准确性。



&emsp;计算机视觉通常趋向于更大更深的网络[6] [18] [17]。 更好的性能通常取决于训练更大的网络或将多个模型组合在一起。 但是，对于  $YOLOv2$  ，我们需要一个更精确的检测器，而且保持很快的速度。 我们不是要扩大网络，而是简化网络，然后让表征(*即目标特征*)更易于学习。 我们将以往工作中的各种创意与我们自己新颖的方法结合起来，以提高 $YOLO$ 的性能。 结果汇总见 表2。



&emsp;**批量标准化**（Batch Normalization）。批量标准化可以显着改善收敛性，而且不再需要其他形式的正则化[7]。 通过在 $YOLO$ 中的所有卷积层上添加批量标准化，可以在 $mAP$  中获得 2％以上的改进。 批量标准化也有助于规范模型。 通过批量标准化，可以从模型中删除 $dropout$ 而不会发生过拟合。


&emsp;**高分辨率分类器** （High Resolution Classifier）。所有的最先进的检测方法都使用在 $ImageNet$ 上预先训练好的分类器[16]。 从AlexNet开始，大多数分类器用小于 $256×256$ 的图像作为输入[8]。 最初的 $YOLO$ 以 $224×224$ 的图像训练分类器网络，并将分辨率提高到 $448$ 以进行检测训练。 这意味着网络必须切换到目标检测的学习，同时能调整到新的输入分辨率。



&emsp;对于  $YOLOv2$  ，我们首先以 $448×448$ 的全分辨率在 $ImageNet$ 上进行 $10$ 个迭代周期的微调。这给予网络一些时间，以调整其滤波器来更好地处理更高分辨率的输入。然后，我们再对该检测网络进行微调。 这个高分辨率的分类网络使 $mAP$  增加了近4％。



&emsp; **卷积与锚框**。 $YOLO$ 直接使用卷积特征提取器顶部的全连接层来预测边界框的坐标。而 $Fast \ R-CNN$ 不是直接预测坐标，是使用手工选取的先验来预测边界框[15]。而不是直接预测坐标 $Fast \ R-CNN$ 预测边界框使用手工挑选的先验区域[15]。 Faster R-CNN中的候选区域生成网络（RPN）仅使用卷积层来预测锚框的偏移和置信度。由于预测层是卷积的，所以RPN可以在特征图中的每个位置预测这些偏移。使用预测偏移代替坐标，可以简化问题并使网络更易于学习。



&emsp;我们从 $YOLO$ 中移除全连接层，并使用锚框来预测边界框。 首先我们消除一个池化层，以使网络卷积层的输出具有更高的分辨率。 我们还缩小网络，使其在分辨率为 $416X416$ 的输入图像上运行，而不是 $448×448$ 。我们这样做是因为我们想要在特征图中有奇数个位置，从而有一个单一的中心单元格。目标，尤其是大的目标，往往占据图像的中心，所以最好在正中心拥有单独一个位置来预测这些目标，而不是在中心附近的四个位置。 $YOLO$ 的卷积层对图像进行了 $32$ 倍的采样，所以通过使用 $416$ 的输入图像，我们得到 $13×13$ 的输出特征图。

&emsp;当我们移动到锚框时，我们将类预测机制与空间位置分开处理，单独预测每个锚框的类及其目标。 遵循原来的 $YOLO$ 的做法，目标预测依然预测了真实标签框（ground truth box）和候选框的 $IOU$ ，而类别预测也是预测了当有目标存在时，该类别的条件概率。



&emsp;使用锚框，精度值会小幅下降。因为原始的 $YOLO$ 仅为每个图片预测98个框，但使用锚框后，我们的模型预测的框数超过 1000 个。 在没有锚框情况下，我们的中等模型将获得 69.5 的 $mAP$  ，召回率为81％。 使用锚框后，我们的模型获得了$69.2$ 的 $mAP$  ，召回率为88％。尽管 $mAP$  减少，但召回率的增加意味着我们的模型有更大的改进空间。

<p> 
<!-- <img src = "8_yolo_v1-v7/02_yolo_imgs/图片02.png"> -->
<img src ="https://oneflow-static.oss-cn-beijing.aliyuncs.com/one-yolo/document/02_yolo_imgs/%E5%9B%BE%E7%89%8702.png">

图2：$VOC$ 和 $ $COCO$ $ 上的聚类框尺寸。我们在边界框的维上运行 $k-means$ 聚类，以获得我们模型的良好先验。左图显示了我们通过k的各种选择获得的平均 $IOU$ 。我们发现 $k = 5$ 为召回与模型的复杂性提供了良好的折中。右图显示了 $VOC$ 和 $COCO$ 的相对质心。这两种方案都喜欢更薄，更高的框，并且 $COCO$ 的尺寸的变化比 $VOC$ 更大。

$k-means$算法: $K-means$ 算法是很典型的基于距离的聚类算法，采用距离作为相似性的评价指标，即认为两个对象的距离越近，其相似度就越大。该算法认为簇是由距离靠近的对象组成的，因此把得到紧凑且独立的簇作为最终目标。
</p>

&emsp;**维度聚类**（Dimension Clusters）。当把锚框与YOLO一起使用时，我们会遇到两个问题。 首先是框的尺寸是手工挑选的。虽然网络可以通过学习适当地调整方框，但是如果我们从一开始就为网络选择更好的先验框，就可以让网络更容易学习到更好的检测结果。



&emsp;我们不用手工选择先验框，而是在训练集的边界框上运行k-means聚类，自动找到良好的先验框。 如果我们使用具有欧几里得距离的标准 $k-means$ ，那么较大的框比较小的框产生更多的误差。 然而，我们真正想要的是独立于框的大小的，能获得良好的 $IOU$ 分数的先验框。 因此对于距离度量我们使用:

<center>

$d(\text { box, centroid }) = 1-\operatorname{IOU}(\text { box }, \text { centroid })$

</center>

&emsp;我们用不同的 $k$ 值运行 $k-means$ ，并绘制最接近质心的平均 $IOU$（见图2）。为了在模型复杂度和高召回率之间的良好折衷，我们选择 $k = 5$。聚类的质心与手工选取的锚框显着不同，它有更少的短且宽的框，而且有更多既长又窄的框。


&emsp;表1中，我们将聚类策略的先验框中心数和手工选取的锚框数在最接近的平均 $IOU$ 上进行比较。仅5个先验框中心的平均 $IOU$ 为61.0，其性能类似于9个锚框的60.9。 使用9个质心会得到更高的平均 $IOU$ 。这表明使用 $k-means$ 生成边界框可以更好地表示模型并使其更容易学习。

<p>

$\begin{array}{lcc}
\text { Box Generation } & \# & \text { Avg IOU } \\
\hline \text { Cluster SSE } & 5 & 58.7 \\
\text { Cluster IOU } & 5 & 61.0 \\
\text { Anchor Boxes [15] } & 9 & 60.9 \\
\text { Cluster IOU } & 9 & 67.2
\end{array}$

表1： $VOC \  2007$ 最接近先验的框的平均 $IOU$。 $VOC \  2007$ 上的目标的平均IOU与其最接近的，未经修改的使用不同生成方法的目标之间的平均 $IOU$ 。聚类得结果比使用手工选取的先验框结果要好得多。

</p>

&emsp; **直接位置预测**（Direct location prediction）。当在 $YOLO$ 中使用锚框时，我们会遇到第二个问题：模型不稳定，尤其是在早期迭代的过程中。 大多数不稳定来自于预测框的 $(x,y)$ 位置。 在候选区域网络中，网络预测的 $t_x,t_y$  ，和中心坐标 $(x,y)$ 计算如下：

<center>

$\huge\begin{array}{l}
x=\left(t_{x} * w_{a}\right)-x_{a} \\
y=\left(t_{y} * h_{a}\right)-y_{a}
\end{array}$

</center>

&emsp;例如，预测 $t_x = 1$ 会使该框向右移动锚框的宽度，而预测 $t_x = -1$ 会将其向左移动相同的宽度。



&emsp;这个公式是不受约束的，所以任何锚框都可以在图像中的任何一点结束，而不管锚框是在哪个位置预测的。随机初始化模型需要很长时间才能稳定到预测合理的偏移量。



&emsp;我们没有预测偏移，而是遵循 $YOLO$ 的方法，预测相对于网格单元位置的位置坐标。这使得真实值的界限在0到1之间。我们使用逻辑激活来限制网络的预测落在这个范围内。



&emsp;网络为特征图的输出的每个单元预测5个边界框。网络预测每个边界框的5个坐标$t_x,t_y,t_w,t_h和t_o$ 。如果单元格从图像的左上角偏移了,并且之前的边界框具有宽度和高度 $p_w,p_h$ 则预测对应于：

<center>

$\begin{aligned}
b_{x} &=\sigma\left(t_{x}\right)+c_{x} \\
b_{y} &=\sigma\left(t_{y}\right)+c_{y} \\
b_{w} &=p_{w} e^{t_{w}} \\
b_{h} &=p_{h} e^{t_{h}} \\
\operatorname{Pr}(\text { object }) * \operatorname{IOU}(b, \text { object }) &=\sigma\left(t_{o}\right)
\end{aligned}$

</center>


&emsp;由于我们限制了位置预测，使得参数化更容易学习，从而使网络更加稳定。使用维度集群以及直接预测边界框中心位置，可以使 $YOLO$ 比锚框的版本提高近5％。


<p align = "center">
<!-- <img src = "8_yolo_v1-v7/02_yolo_imgs/图片03.png"> -->
<img src = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/one-yolo/document/02_yolo_imgs/%E5%9B%BE%E7%89%8703.png">

图3：具有维度先验和位置预测的边界框。我们预测框的宽度和高度作为聚类质心的偏移量。我们使用sigmoid函数预测相对于滤波器应用位置的框的中心坐标。
</p>

&emsp;**细粒度功能**（Fine-Grained Features）。修改后的YOLO在 $13×13$ 特征图上预测检测结果。 虽然这对于大型物体是足够的，但使用更细粒度特征对定位较小物体有好处。Faster R-CNN和SSD都在网络中的各种特征图上运行网络，以获得多个分辨率。 我们采取不同的方法，只需添加一个直通层，以 $26×26$ 的分辨率从较早的层中提取特征。



&emsp;直通层将高分辨率特征与低分辨率特征连接起来，将相邻特征叠加到不同的通道中，而不是空间位置上，类似于 $ResNet$ 中的恒等映射。将 $26×26×512$ 的特征图变为 $13×13×2048$ 的特征图，然后就可以与原来的特征连接。我们的检测器运行在这张扩展的特征图的顶部，以便它可以访问细粒度的功能。这使性能提高了1％。



&emsp;**多尺度训练**（Multi-Scale Training）。原来的 $YOLO$ 使用 $448×448$ 的输入分辨率。通过添加锚框，我们将分辨率更改为 $416×416$ 。但是，由于我们的模型仅使用卷积层和池化层，因此可以实时调整大小。我们希望  $YOLOv2$  能够在不同尺寸的图像上运行，因此我们可以将多尺度训练应到模型中。



&emsp;我们不需要修改输入图像大小，而是每隔几次迭代就改变一次网络。每 $10$ 个批次我们的网络会随机选择一个新的图像尺寸大小。由于我们的模型缩减了 $32$ 倍，所以我们从 $32$ 的倍数中抽取：${320,352，…，608}$ 。因此，最小的选项是 $320×320$ ，最大的是 $608×608$ 。我们调整网络的尺寸到那个维度并继续训练。



&emsp;这个机制迫使网络学习如何在各种输入维度上做好预测。这意味着同一个网络可以预测不同分辨率下的检测结果。网络在较小的尺寸下运行速度更快，因此  $YOLOv2$  在速度和准确性之间提供了一个轻松的折中。


&emsp; 在低分辨率下，  $YOLOv2$  作为一种便宜但相当准确的检测器工作。 在 $288×288$ 情况下，它的运行速度超过 90 FPS，而 $mAP$  几乎与 $Fast \ R-CNN$ 一样好。这使其成为小型 $GPU$ ，高帧率视频或多视频流的理想选择。


&emsp;在高分辨率下，  $YOLOv2$  是一款先进的检测器，在VOC2007上获得了78.6的 $mAP$  ，同时仍以高于实时速度运行。请参阅表3，了解  $YOLOv2$  与其他框架在 $VOC \  2007$ 上的比较 图4。

<p align = "center">
<!-- <img src ="8_yolo_v1-v7/02_yolo_imgs/图片04.png"> -->
<img src = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/one-yolo/document/02_yolo_imgs/%E5%9B%BE%E7%89%8704.png">

图4： $VOC \  2007$ 上的精度和速度
</p>

&emsp;**进一步的实验**（Further Experiments）。 我们在 $VOC \ 2012$ 上训练了 $YOLOv2$  进行检测。表4 显示了  $YOLOv2$  与其他最先进的检测系统的性能比较。   $YOLOv2$  运行速度远高于对手，且精度达到 73.4 $mAP$  。 我们还在 $COCO$ 上训练，并与表5中的其他方法进行比较。使用 $VOC$ 度量（ $IOU = 0.5$ ），  $YOLOv2$  获得44.0  $mAP$  ，与 $SSD$ 和 $Faster \ R-CNN$ 相当。


<center>

$\begin{array}{lrrr}
\text { Detection Frameworks } & \text { Train } & \text { mAP } & \text { FPS } \\
\hline \text { Fast R-CNN [5] } & 2007+2012 & 70.0 & 0.5 \\
\text { Faster R-CNN VGG-16[15] } & 2007+2012 & 73.2 & 7 \\
\text { Faster R-CNN ResNet[6] } & 2007+2012 & 76.4 & 5 \\
\text { YOLO [14] } & 2007+2012 & 63.4 & 45 \\
\text { SSD300 [11] } & 2007+2012 & 74.3 & 46 \\
\text { SSD500 [11] } & 2007+2012 & 76.8 & 19 \\
\hline \text { YOLOv2 288 } \times 288 & 2007+2012 & 69.0 & 91 \\
\text { YOLOv2 352 } \times 352 & 2007+2012 & 73.7 & 81 \\
\text { YOLOv2 416 } \times 416 & 2007+2012 & 76.8 & 67 \\
\text { YOLOv2 480 } \times 480 & 2007+2012 & 77.8 & 59 \\
\text { YOLOv2 } 544 \times 544 & 2007+2012 & \mathbf{7 8 . 6} & 40
\end{array}$

表3： $PA S C A L  \ VOC \ 2007$ 的检测框架。  $YOLOv2$  比以前的检测方法更快，更准确。它也可以以不同的分辨率运行，以便在速度和准确性之间轻松折衷。每个  $YOLOv2$  项实际上都是具有相同权重的相同训练模型，只是以不同的大小进行评估。所有的时间的测试都运行在Geforce GTX Titan X（原始的，而不是Pascal模型）

</center>


## 3.更快
&emsp;我们希望检测结果准确，但我们也希望检测速度更快。 大多数用于检测的应用程序（如机器人或自动驾驶汽车）都依赖于低延迟预测。 为了最大限度地提高性能，我们将  $YOLOv2$   设计从头到尾都非常快 。

<center>

<!-- <img src = "02_yolo_imgs/table03.png"> -->
<img src ="https://oneflow-static.oss-cn-beijing.aliyuncs.com/one-yolo/document/02_yolo_imgs/table03.png">

表2：从 $YOLO$ 到  $YOLOv2$  的路径。大多数列出的设计决策都会导致 $MAP$  显着增加。有两个例外情况是：切换到带有锚框的全卷积网络和使用新网络。切换到锚框方法增加召回率，而不改变 $mAP$  ，而使用新网络削减33％的计算。
</center>

<center>
<!-- <img src = "02_yolo_imgs/table04.png"> -->
<img src = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/one-yolo/document/02_yolo_imgs/table04.png"> 

表4：PASCAL VOC2012测试检测结果。  $YOLOv2$  与采用ResNet和SSD512的Faster R-CNN等先进检测器性能相当，速度提高2至10倍。
</center>

&emsp;大多数检测框架依赖于VGG-16作为基本特征提取器[17]。 $VGG-16$ 是一个功能强大，准确的分类网络，但它有不必要的复杂度。 $VGG-16$ 的卷积层在一个 $224×224$ 分辨率单个图像上运行一次需要 $306.90$ 亿浮点运算。


&emsp; $YOLO$ 框架使用基于 $Googlenet$ 架构的自定义网络[19]。这个网络比 $VGG-16$ 更快，一次前向传播只要 $85.2$ 亿次运行。然而，它的准确性略低于 $VGG-16$。在 $ImageNet$ 上，用 $224×224$ 的单张裁剪图像， $YOLO$ 的自定义模型的精度为88.0％而 $VGG-16$ 则为90.0％。

&emsp; $Darknet-19$ 。我们提出了一个新的分类模型作为  $YOLOv2$  的基础。我们的模型建立在网络设计的先前工作以及该领域的常识上。与 $VGG$ 模型类似，我们大多使用 $3×3$ 滤波器，并且在池化层步骤后使用两倍的通道数[17]。按照Network in Network（NIN）的方法，我们使用全局平均池化来做预测，并使用1×1滤波器来压缩3×3卷积的特征表示[9]。我们使用批量归一化来稳定训练，加速收敛，并规范模型[7]。



&emsp;最终的模型叫做Darknet-19，它有19个卷积层和5个Maxpool层。 $Darknet-19$ 只需要55.8亿次操作来处理图像，但在 $ImageNet$ 上实现了72.9％的top-1精度和91.2％的top-5精度。



&emsp; **分类训练**（Training for classification）。我们使用 $DarkNet$ 神经网络框架，使用随机梯度下降，初始学习率为0.1，多项式速率衰减为4，权重衰减为0.0005，动量为0.9，在标准 $ImageNet$  1000类别分类数据集上对网络进行160个迭代周期的训练[13]。在训练过程中，我们使用标准数据增强技巧，包括随机截取，旋转和改变色相，饱和度和曝光。



&emsp; 如上所述，在我们对 $224×224$ 图像进行初始训练之后，我们用更大的分辨率448对网络进行了微调。微调时，我们使用上述参数进行训练，但仅用10个周期，并且开始时的学习率为10-3。在这个更高的分辨率下，我们的网络实现了76.5％的 $top-1$ 精度和93.3％的 $top-5$ 精度。



&emsp;**检测训练**（Training for detection）。我们这样修改网络：去除最后一个卷积层，取而代之的是添加三个 $3 × 3$ 的卷积层，每个层有 $1024$ 个过滤器，然后在最后添加 $1×1$ 卷积层，该层的滤波器数量是检测需要的输出数量。 对于 $VOC$ ，我们预测 $05$ 个边界框，每个边界框有 5个坐标和20个类别，所以有125个滤波器。我们还添加了从最后的 $3×3×512$ 层到倒数第二层卷积层的直通层，以便我们的模型可以使用细粒度特征。


&emsp;我们训练网络160个迭代周期，初始学习率为 $10^{-3}$，在60和90周期同时除以10。我们使用 $0.0005$ 的权值衰减和 0.9 的动量(*momentum*)。我们对 $YOLO$ 和 $SSD$ 进行类似的数据增强，随机裁剪，色彩修改等。我们在 $COCO$ 和 $VOC$ 使用相同的训练策略。


<center>

$\begin{array}{l|c|cccc|ccc|ccc|ccc} 
& & 0.5: 0.95 & 0.5 & 0.75 & \mathrm{~S} & \mathrm{M} & \mathrm{L} & 1 & 10 & 100 & \mathrm{~S} & \mathrm{M} & \mathrm{L} \\
\hline \text { Fast R-CNN [5] } & \text { train } & 19.7 & 35.9 & - & - & - & - & - & - & - & - & - & - \\
\text { Fast R-CNN[1] } & \text { train } & 20.5 & 39.9 & 19.4 & 4.1 & 20.0 & 35.8 & 21.3 & 29.5 & 30.1 & 7.3 & 32.1 & 52 .0 \\
\text { Faster R-CNN[15] } & \text { trainval } & 21.9 & 42.7 & - & - & - & - & - & - & - & - & - & - \\
\text { ION [1] } & \text { train } & 23.6 & 43.2 & 23.6 & 6.4 & 24.1 & 38.3 & 23.2 & 32.7 & 33.5 & 10.1 & 37.7 & 53.6 \\
\text { Faster R-CNN[10] } & \text { trainval } & 24.2 & 45.3 & 23.5 & 7.7 & 26.4 & 37.1 & 23.8 & 34.0 & 34.6 & 12.0 & 38.5 & 54.4 \\
\text { SSD300 [11] } & \text { trainval35k } & 23.2 & 41.2 & 23.4 & 5.3 & 23.2 & 39.6 & 22.5 & 33.2 & 35.3 & 9.6 & 37.6 & 56.5 \\
\text { SSD512 [11] } & \text { trainval35k } & \mathbf{26.8} & \mathbf{4 6 . 5} & \mathbf{2 7 . 8} & \mathbf{9 . 0} & \mathbf{2 8 . 9} & 41.9 & \mathbf{2 4 . 8} & 37.5 & \mathbf{3 9 . 8} & \mathbf{1 4 . 0} & 43.5 & 59.0 \\
\hline \text { YOLOv2 [11] } & \text { trainval35k } & 21.6 & 44.0 & 19.2 & 5.0 & 22.4 & 35.5 & 20.7 & 31.6 & 33.3 & 9.8 & 36.5 & 54 .4
\end{array}$

表5： $COCO$  test-dev集上的结果，来源于论文[11]
</center>

## 4.更强
&emsp;我们提出了一个联合训练分类和检测数据的机制。 我们的方法使用了用于检测的图像来学习检测特定信息，如边界框坐标预测和目标以及如何对常见目标进行分类。通过使用仅具有类标签的图像来扩展其可检测类别的数量。



&emsp;在训练期间，我们混合来自检测和分类数据集的图像。 当我们的网络看到标记为检测的图像时，可以根据完整的  $YOLOv2$  损失函数进行反向传播。 当它看到分类图像时，只会反向传播分类部分的损失。

<center>

$\begin{array}{l|c|c|c}
\text { Type } & \text { Filters } & \text { Size/Stride } & \text { Output } \\
\hline \text { Convolutional } & 32 & 3 \times 3 & 224 \times 224 \\
\text { Maxpool } & & 2 \times 2 / 2 & 112 \times 112 \\
\text { Convolutional } & 64 & 3 \times 3 & 112 \times 112 \\
\text { Maxpool } & & 2 \times 2 / 2 & 56 \times 56 \\
\text { Convolutional } & 128 & 3 \times 3 & 56 \times 56 \\
\text { Convolutional } & 64 & 1 \times 1 & 56 \times 56 \\
\text { Convolutional } & 128 & 3 \times 3 & 56 \times 56 \\
\text { Maxpool } & & 2 \times 2 / 2 & 28 \times 28 \\
\text { Convolutional } & 256 & 3 \times 3 & 28 \times 28 \\
\text { Convolutional } & 128 & 1 \times 1 & 28 \times 28 \\
\text { Convolutional } & 256 & 3 \times 3 & 28 \times 28 \\
\text { Maxpool } & & 2 \times 2 / 2 & 14 \times 14 \\
\text { Convolutional } & 512 & 3 \times 3 & 14 \times 14 \\
\text { Convolutional } & 256 & 1 \times 1 & 14 \times 14 \\
\text { Convolutional } & 512 & 3 \times 3 & 14 \times 14 \\
\text { Convolutional } & 256 & 1 \times 1 & 14 \times 14 \\
\text { Convolutional } & 512 & 3 \times 3 & 14 \times 14 \\
\text { Maxpool } & & 2 \times 2 / 2 & 7 \times 7 \\
\text { Convolutional } & 1024 & 3 \times 3 & 7 \times 7 \\
\text { Convolutional } & 512 & 1 \times 1 & 7 \times 7 \\
\text { Convolutional } & 1024 & 3 \times 3 & 7 \times 7 \\
\text { Convolutional } & 512 & 1 \times 1 & 7 \times 7 \\
\text { Convolutional } & 1024 & 3 \times 3 & 7 \times 7 \\
\hline \hline \text { Convolutional } & 1000 & 1 \times 1 & 7 \times 7 \\
\text { Avgpool } & & \text { Global } & 1000 \\
\text { Softmax } & & &
\end{array}$

表6：Darknet-19
</center>

&emsp;这种方法带来了一些难题。检测数据集只有常用的目标和通用的标签，如“狗”或“船”。分类数据集具有更广泛和更深入的标签范围。  $ImageNet$ 拥有多种犬种，包括Norfolk terrier，Yorkshire terrier和Bedlington terrier。如果我们想在两个数据集上进行训练，则需要采用一致的方式来合并这些标签。



&emsp;大多数分类方法使用涵盖所有可能类别的 $softmax$ 层来计算最终概率分布。使用 $softmax$ ，意味着类是相互排斥的。这给组合数据集带来了问题，例如，你不能用这个模型来组合 $ImageNet$ 和 $COCO$ ，因为类别 $Norfolk \ terrier和dog$ 不是互斥的。


&emsp;相反，我们可以使用多标签模型来组合不会互相排斥的数据集。这个方法忽略了我们所知道的关于数据的所有结构，例如所有的 $COCO$ 类都是相互独立的。



&emsp;**分层分类** （Hierarchical classification）。  $ImageNet$ 标签是从 $WordNet$ 中提取的，$WordNet$ 是一个构建概念及其相互关系的语言数据库[12]。 Norfolk terrier和Yorkshire terrier都是terrier的下义词，terrier是一种hunting dog，hunting dog是dog，dog是canine等。大多数分类的方法假设标签是一个扁平结构，但是对于组合数据集，结构正是我们所需要的。



&emsp; $WordNet$ 的结构是有向图，而不是树，因为语言很复杂。 例如，“狗”既是一种“犬”又是一种“家养动物”，它们都是 $WordNet$ 中的同义词。 我们不使用完整的图结构，而是通过从 $ImageNet$ 中的概念构建分层树来简化问题。



&emsp;WordNet的结构是有向图，而不是树，因为语言很复杂。例如，一只狗既是一种犬科动物，又是一种家养动物，它们都是WordNet中的同种动物。我们没有使用完整的图结构，而是通过从 $ImageNet$ 中的概念构建分层树来简化问题。


&emsp;为了构建这棵树，我们检查 $ImageNet$ 中的视觉名词，并查看它们通过 $WordNet$ 图到根节点的路径，在这种情况下是“物理目标”。 许多同义词只有在图上一条路径，所以首先我们将所有这些路径添加到我们的树中。 然后，我们反复检查我们留下的概念，并尽可能少地添加生成树的路径。 所以如果一个概念有两条通向根的路径，一条路径会为我们的树增加三条边，另一条路只增加一条边，我们选择较短的路径。



&emsp; 最终的结果是 $WordTree$ ，一个视觉概念的分层模型。为了使用  $WordTree$  进行分类，我们预测每个节点的条件概率，以得到同义词集合中每个同义词下义词的概率。例如，在terrier节点我们预测：

<center>

$\large\operatorname{Pr}  (Norfolk \ terrier|terrier) \\
 \operatorname{Pr}  (Yorkshire \ terrier|terrier) \\
 \operatorname{Pr}(  Bedlington \ terrier|terrier  )$
</center>

&emsp;如果我们想要计算一个特定节点的绝对概率，我们只需沿着通过树到达根节点的路径，再乘以条件概率。所以如果我们想知道一张图片是否是Norfolk terrier，我们计算：

<center>

$\operatorname{Pr}(\text { Norfolk terrier }) = \operatorname{Pr}(\text { Norfolk terrier } \mid
\text { terrier })$

$\quad * \operatorname{Pr}(\text { terrier } \mid \text { hunting dog })$

$* \ldots * \\$

$* \operatorname{Pr}(\text { mammal } \mid \operatorname{Pr}(\text { animal })$

$* \operatorname{Pr}(\text { animal } \mid \text { physical object })$

</center>

&emsp;为了实现分类，我们假定图像包含一个目标：$P r(physical object) = 1$。

&emsp;为了验证这种方法，我们在使用1000类 $ImageNet$ 构建的 $WordTree$  上训练 $Darknet-19$ 模型。 为了构建 $WordTree1k$ ，我们添加了所有中间节点，将标签空间从 $1000$ 扩展到 $1369$。在训练过程中，我我们将真实标签向树上面传播，以便如果图像被标记为Norfolk terrier，则它也被标记为dog和mamal等。为了计算条件概率，我们的模型预测了1369个值的向量，并且我们计算了相同概念的下义词在所有同义词集上的softmax，见图5。



&emsp;使用与以前相同的训练参数，我们的分层Darknet-19达到了71.9％的 $top-1$ 精度和90.4％的 $top-5$ 精度。 尽管增加了369个附加概念，并且让我们的网络预测了树状结构，但我们的精度仅略有下降。 以这种方式进行分类也有若干好处。 在新的或未知的目标类别上，性能会优雅低降低。 例如，如果网络看到一张狗的照片，但不确定它是什么类型的狗，它仍然会以高置信度预测“dog”，只是在下义词会有较低的置信度。

&emsp;该方法也适用于检测。现在，我们不用假定每个图像都有一个目标物体，而是使用  $YOLOv2$  的目标预测器给出P r（目标物体）的值。检测器预测边界框和概率树。我们遍历树，在每次分割中选取具有最高的置信度的路径，直到达到某个阈值，然后我们得到该目标的类别。

&emsp; **数据集与 $WordTree$  的组合** (Dataset combination with WordTree)。我们可以使用 $WordTree$  以可行的方式将多个数据集组合在一起。我们只需将数据集中的类别映射到树中的synsets即可。图6显示了一个使用 $WordTree$  组合来自 $ImageNet$ 和 $COCO$ 的标签的示例。 WordNet非常多样化，因此我们可以将这种技术用于大多数数据集。

<p>

<!-- <img src = "02_yolo_imgs/图片05.png"> -->
<img src = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/one-yolo/document/02_yolo_imgs/%E5%9B%BE%E7%89%8705.png">

图5：对 $ImageNet$ 与 $WordTree$  的预测。大多数ImaNet模型使用一个大的softmax来预测概率分布。使用 $WordTree$  ，我们通过共同的下位词执行多个softmax操作。

</p>

&emsp;**联合分类和检测**(Joint classification and detection)。现在我们可以使用 $WordTree$  组合数据集，在分类和检测上训练联合模型。我们想要训练一个非常大规模的检测器，所以使用 $COCO$ 检测数据集和完整 $ImageNet$ 版本中的前9000类创建我们的组合数据集。我们还需要评估我们的方法，以便从 $ImageNet$ 检测挑战中添加任何尚未包含的类。该数据集的相应 $WordTree$  具有9418个类。 $ImageNet$ 有更大的数据集，所以我们通过对 $COCO$ 进行过采样来平衡数据集，使得 $ImageNet$ 与 $COCO$ 的比例略大于 $4:1$。



&emsp;我们使用上述的数据集训练 $YOLO9000$ 。 我们使用基本的  $YOLOv2$  架构，但只有3个先验框而不是5个来限制输出大小。当我们的网络处理检测图像时，我们会像平常一样反向传播损失。对于分类损失，我们只是将损失反向传播到标签相应级别或更高的级别。 例如，如果标签是狗，我们不会将任何错误给树做进一步预测，如德国牧羊犬与黄金猎犬，因为我们没有这些信息。


<p>
<!-- <img src = "02_yolo_imgs/图片06.png"> -->
<img src = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/one-yolo/document/02_yolo_imgs/%E5%9B%BE%E7%89%8705.png">

图6：使用 $WordTree$  层次结构组合数据集。使用WordNet概念图，我们构建了视觉概念的分层树。然后，我们可以通过将数据集中的类映射到树中的synsets来合并数据集。出于说明目的，这是 $WordTree$  的简化视图。
</p>

&emsp;当网络处理分类图像时，我们只是反向传播分类损失。要做到这一点，我们只需找到预测该类别最高概率的边界框，然后在预测的树上计算损失。我们还假设预测框与真实框的 $IOU$ 至少为0.3，并且基于这个假设我们反向传播目标损失。



&emsp;通过这种联合训练， $YOLO9000$ 学习使用 $COCO$ 中的检测数据来查找图像中的目标，并学习使用来自 $ImageNet$ 的数据对各种这些目标进行分类。



&emsp;我们在 $ImageNet$ 检测任务上评估 $YOLO9000$ 。  $ImageNet$ 的检测任务与 $COCO$ 共享44个目标类别，这意味着 $YOLO9000$ 看到的测试图像大多数是分类数据，而不是检测数据。  $YOLO9000$ 的总 $mAP$  是19.7  $mAP$  ，其中在不相交的156个目标类上， $YOLO9000$ 从未见过这些类的任何检测数据的标签，仍获得了16.0 $mAP$  。这个 $mAP$  高于DPM的结果，但 $YOLO9000$ 是在部分监督[4]的不同的数据集上训练的。而且它能同时检测9000个其他目标类别，所有的检测都是实时的。



&emsp;在分析 $YOLO9000$ 在 $ImageNet$ 上的表现时，我们发现它很好地学习了新的动物种类，但是在像服装和设备这样的学习类别中表现不佳。新动物更容易学习，因为目标预测可以从 $COCO$ 中的动物泛化的很好。相反， $COCO$ 没有任何类型的衣服的边界框标签，只针对人，因此 $YOLO9000$ 在分类“墨镜”或“泳裤”等类别上存在困难。


<center>

$\begin{array}{ll}
\text { diaper } & 0.0 \\
\text { horizontal bar } & 0.0 \\
\text { rubber eraser } & 0.0 \\
\text { sunglasses } & 0.0 \\
\text { swimming trunks } & 0.0 \\
\ldots & \\
\text { red panda } & 50.7 \\
\text { fox } & 52.1 \\
\text { koala bear } & 54.3 \\
\text { tiger } & 61.0 \\
\text { armadillo } & 61.7
\end{array}$

表7： $ImageNet$ 上的 $YOLO9000$ 最佳和最差类别。 156个弱监督类的AP最高和最低的类。  $YOLO9000$ 模型很好地预测各种各样的动物，但不擅长预测诸如服装或设备等的新类。
</center>

## 5.总结
&emsp;我们介绍实时检测系统  $YOLOv2$  和 $YOLO9000$ 。   $YOLOv2$  在各种检测数据集中都是最先进的，并且比其他检测系统更快。此外，它可以在各种图像尺寸下运行，以提供速度和准确性之间的平滑折中。



 &emsp;$YOLO9000$ 是一个通过联合优化检测和分类来检测超过9000个目标类别的实时框架。我们使用 $WordTree$  将各种来源的数据和我们的联合优化技术相结合，在 $ImageNet$ 和 $COCO$ 上同时进行训练。  $YOLO9000$ 向缩小检测和分类之间的数据集大小的差距迈出了坚实的一步。



&emsp;我们的许多技术都是泛化到目标检测之外的领域。  $ImageNet$ 的 $WordTree$  表示方法为图像分类提供了更丰富，更详细的输出空间。使用分层分类的数据集组合在分类和分割领域将会很有用。像多尺度训练这样的训练技术可以为各种视觉任务提供帮助。



&emsp;对于未来的工作，我们希望使用类似的技术进行弱监督图像分割。我们还计划使用更强大的匹配策略来改善我们的检测结果，以在训练期间将弱标签分配给分类数据。计算机视觉拥有大量的标记数据。我们将继续寻找方法，将不同的数据来源和数据结构结合在一起，形成更强大的视觉世界模型。

## References

- [1]	S. Bell, C. L. Zitnick, K. Bala, and R. Girshick. Inside-outside net: Detecting objects in context with skip pooling and recurrent neural networks. arXiv preprint arXiv:1512.04143, 2015. 6
- [2]	J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on, pages 248–255. IEEE, 2009. 1
- [3]	M. Everingham, L. Van Gool, C. K. Williams, J. Winn, and
A.	Zisserman. The pascal visual object classes (voc) chal-lenge. International journal of computer vision, 88(2):303–338, 2010. 1
- [4]	P. F. Felzenszwalb, R. B. Girshick, and D. McAllester. Discriminatively trained deformable part models, release 4. http://people.cs.uchicago.edu/ pff/latent-release4/. 8
- [5]	R. B. Girshick. Fast R-CNN. CoRR, abs/1504.08083, 2015. 4, 5, 6
- [6]	K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learn-ing for image recognition. arXiv preprint arXiv:1512.03385, 2015. 2, 4, 5
- [7]	S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015. 2, 5
- [8]	A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pages 1097–1105, 2012. 2
- [9]	M. Lin, Q. Chen, and S. Yan. Network in network. arXiv preprint arXiv:1312.4400, 2013. 5
- [10]	T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ra-manan, P. Doll´ar, and C. L. Zitnick. Microsoft coco: Com-mon objects in context. In European Conference on Com-puter Vision, pages 740–755. Springer, 2014. 1, 6
- [11]	W. Liu, D. Anguelov, D. Erhan, C. Szegedy, and S. E. Reed. SSD: single shot multibox detector. CoRR, abs/1512.02325, 2015. 4, 5, 6
- [12]	G. A. Miller, R. Beckwith, C. Fellbaum, D. Gross, and K. J. Miller. Introduction to wordnet: An on-line lexical database. International journal of lexicography, 3(4):235–244, 1990. 6
- [13]	J. Redmon. Darknet: Open source neural networks in c. http://pjreddie.com/darknet/, 2013–2016. 5
- [14]	J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. You only look once: Unified, real-time object detection. arXiv preprint arXiv:1506.02640, 2015. 4, 5
- [15]	S. Ren, K. He, R. Girshick, and J. Sun. Faster r-cnn: To-wards real-time object detection with region proposal net-works. arXiv preprint arXiv:1506.01497, 2015. 2, 3, 4, 5, 6
- [16] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 2015. 2
- [17] K. Simonyan and A. Zisserman. V ery deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 2, 5
- [18] C. Szegedy, S. Ioffe, and V . V anhoucke. Inception-v4, inception-resnet and the impact of residual connections on learning. CoRR, abs/1602.07261, 2016. 2 
- [19] C. Szegedy, W. Liu, Y . Jia, P . Sermanet, S. Reed, D. Anguelov, D. Erhan, V . V anhoucke, and A. Rabinovich.Going deeper with convolutions. CoRR, abs/1409.4842, 2014. 5 
- [20] B. Thomee, D. A. Shamma, G. Friedland, B. Elizalde, K. Ni, D. Poland, D. Borth, and L.-J. Li. Yfcc100m: The new data in multimedia research. Communications of the ACM, 59(2):64–73, 2016. 1