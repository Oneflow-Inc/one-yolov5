FROM nvcr.io/nvidia/pytorch:23.03-py3

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt update && apt install -y libopenblas-dev nasm python3-pip autoconf libtool autotools-dev libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libjpeg-dev libfreetype6-dev zlib1g-dev libpng-dev libfontconfig1 && apt clean

# install oneflow
RUN python3 -m pip install --pre oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/release/yolo_demo/cu118

# download pretrained models and dataset
WORKDIR /code
RUN mkdir -p /code/data && cd /code/data && \
    wget http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/yolov5x-seg.pt && \
    wget http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/yolov5s-seg.pt && \
    wget http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/coco128-seg.zip && \
    unzip coco128-seg.zip && rm coco128-seg.zip

WORKDIR /code
RUN git clone https://github.com/Oneflow-Inc/one-yolov5.git && \
    cd /code/one-yolov5 && git checkout proj_demo && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache/pip

