FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
RUN apt update && apt install -y libopenblas-dev nasm g++ gcc python3-pip cmake autoconf libtool git wget vim autotools-dev libgl1-mesa-dev libglib2.0-0

WORKDIR /workspace
RUN git clone https://github.com/Oneflow-Inc/oneflow.git && git clone https://github.com/Oneflow-Inc/one-yolov5.git
RUN cd /workspace/oneflow && \
    git checkout 5ab28bbca && \
    mkdir build && cd build && \
    cmake .. -C ../cmake/caches/cn/cuda.cmake && \
    make -j$(nproc) && source source.sh   

RUN mkdir /workspace/data && cd /workspace/data && \
    wget http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/yolov5x-seg.pt && \
    wget http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/yolov5s-seg.pt && \
    wget http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/coco128-seg.zip && \
    unzip coco128-seg.zip && rm coco128-seg.zip

RUN cd /workspace/one-yolov5 && git checkout proj_demo && \
    pip install -r requirements.txt

