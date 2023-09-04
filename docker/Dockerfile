FROM nvcr.io/nvidia/pytorch:23.03-py3

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
RUN apt update && apt install -y libopenblas-dev nasm g++ gcc python3-pip cmake autoconf libtool git wget vim unzip autotools-dev libgl1-mesa-dev libglib2.0-0

WORKDIR /code
RUN git clone https://github.com/Oneflow-Inc/oneflow.git && git clone https://github.com/Oneflow-Inc/one-yolov5.git
RUN cd /code/oneflow && \
    git checkout 5ab28bbca && \
    mkdir build && cd build && \
    cmake .. -C ../cmake/caches/cn/cuda.cmake && \
    make -j$(nproc)
ENV PYTHONPATH=/code/oneflow/python:$PYTHONPATH

RUN mkdir /code/data && cd /code/data && \
    wget http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/yolov5x-seg.pt && \
    wget http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/yolov5s-seg.pt && \
    wget http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/coco128-seg.zip && \
    unzip coco128-seg.zip && rm coco128-seg.zip

#RUN apt update && apt-get install -y libsm6 libxrender1 libfreetype6 libfontconfig1 libfreetype6-dev
RUN apt update && apt-get install -y libsm6 libxrender1 libjpeg-dev libfreetype6-dev zlib1g-dev libpng-dev libfontconfig1

RUN cd /code/one-yolov5 && git checkout proj_demo && \
    pip install -r requirements.txt && \
    pip install -U smmap gitdb gitpython pillow && \
    rm -rf /root/.cache/pip

