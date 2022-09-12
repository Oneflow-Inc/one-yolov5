
# 一 one-yolov5特色

- one-yolo实现了什么？
1. 以oneflow为后端移植了最火的 ultralytics/yolov5，在完全匹配pytorch api的同时达到不低于pytorch训练精度的同时拥有更好的训练速度，可以缩短模型开发周期。（调优还在进行中）
2. 我们将ultralytics/yolov5的相关训练教程进行了中文化，可以在工程提供的文档网站上查看。
3. 我们制作了一个《从零开始学会yolov5》电子书，通过逐行代码解析的方式揭开yolov5的所有细节。电子书目前的目录是：
     - 1. YOLOv5 网络结构解析（[已完成](https://github.com/Oneflow-Inc/one-yolov5/pull/22/files#diff-8ff42d2e1cf4be6aaa21912d441ac6b0ddc7fb350d0b9504f635c56f36cdfdf0) 在这里可以看到）
     - 2. 如何准备yolov5模型训练数据(以coco数据集和自定义数据集为例) （梳理了一半，其它章节正在开发 by 冯文）
     - 3. Model Train(以coco数据集为例)
     - 4. YOLOv5的数据组织与处理源码解读
     - 5. YOLOv5中Loss部分计算
     - 6. 模型导出和部署介绍（正在开发和梳理 by BBuf）
     - 7. 网页部署和app。
     - 8. 和tvm的交互，基于tvm的部署。（fp32+int8量化 联动tvm oneflow前端吸引部分人关注）
     - 9. YOLOv5中的参数搜索
     - 10. torch_utils/ 文件夹下的其它trick介绍，还可以做2-3期。
 4. 基于这份电子书我们会在b站同步制作一系列教程讲解yolov5的细节，天然植入oneflow。电子书的每一节都会以知乎文章以及博客/公众号文章的方式发表。



![alt](https://2486075003-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-M6S9nPJhEX9FYH6clfW%2Fuploads%2FfHpPTWNdCVR9qHQDeskF%2FScreen%20Shot%202022-08-24%20at%2012.35.36%20PM.png?alt=media&token=623927fe-3099-4ccd-8aaa-890bf5c0b03b)

- 学习本教程能够收获什么?

# 二 YOLOv5 网络结构解析
1. 模型配置文件解读
2. 模型
   
# 三 如何准备yolov5模型训练数据(以coco数据集和自定义数据集为例)
1. 测试数据集的结构(coco评测标准)
2. 如何自制测试数据集

# 四 Model Train(以coco数据集为例)
1. 项目结构介绍
2. 训练结果使用coco数据集
3. 数据输入
4. 输出()
5. 结果解析包括 mAP, precision, and recall。
6. 模型测试 val.py 
7. 效果演示 detect.py 

# 五 YOLOv5的数据组织与处理源码解读
1. yolov5 的 mosaic 数据增强。
2. Rectangular inference（矩形推理）


# 六 YOLOv5中Loss部分计算

# 七 网络模型的构建源码解读

# 八 模型导出 

# 九 YOLOv5中的参数搜索
