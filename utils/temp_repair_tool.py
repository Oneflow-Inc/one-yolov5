# One-YOLOv5 🚀 fix by OneFlow
"""
缺少：
1. https://pytorch.org/docs/stable/mobile_optimizer.html?highlight=optimize_for_mobile#torch.utils.mobile_optimizer.optimize_for_mobile

"""
"""
未对齐:
1. i = flowvision.ops.nms(boxes, scores, iou_thres)  # NMS flow.nms()
2. imgs = imgs.to(device).float 
"""

import os
import psutil
import subprocess
import threading
import time
import threading
import oneflow as flow
import numpy as np
from pathlib import Path


class FlowCudaMemoryReserved:
    """实现思路
    在 Python 的类中起一个子线程，并让子线程每 _update_time 秒更新一次数据.
    子线程 根据 pid 更新显存。
    O(1) 获取显存占用
    """

    def __init__(self, device_type="GPU") -> None:
        self._device_type = device_type
        self._pid = str(os.getpid())
        self._memery_of_pid = "None"
        self._update_time = 3.0
        thread = threading.Thread(target=self.update_data, args=())
        thread.daemon = True
        thread.start()
        print(f"current pid {self._pid=}")

    def cuda_memory_reserved(self, mode="MB"):
        memory = self._memery_of_pid
        try:
            if mode == "MB":
                return "%.3fMB" % (float(memory))
            elif mode == "GB":
                return "%.3fG" % (float(memory) / 1024)
            elif mode not in ("MB", "GB"):
                print(f"warning {mode=}")
                return None
            else:
                return "None"
        except:
            # print(f'warning cuda_memory_reserved')
            return "None"

    def update_data(self):
        while True:
            self._memery_of_pid = self.use_memory(mode="MB")
            time.sleep(self._update_time)

    def use_memory(self, mode="MB"):
        pid = self._pid
        try:
            memory = (
                self.get_gpu_memory(pid)
                if self._device_type == "GPU"
                else self.get_cpu_memory(pid)
            )
            if mode == "MB":
                return "%.3f" % (float(memory))
            else:
                return "%.3f" % (float(memory) / 1024)
        except:
            return "None"

    def get_cpu_memory(self, pid):
        process = psutil.Process(pid)
        memory_info = process.memory_info()
        memory = memory_info.rss / 1024 ** 2  # MB
        return memory

    def get_gpu_memory(self, pid):
        process_memory = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"]
        )
        process_memory_list = process_memory.strip().decode("utf-8").split("\n")
        for process_memory in process_memory_list[1:]:
            process_memory = process_memory.split(",")
            current_pid = process_memory[0]
            if current_pid == pid:
                current_memory = process_memory[1].strip().split(" ")
                return current_memory[0]
        return None




def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {
        k: v
        for k, v in da.items()
        if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape
    }


def load_pretrained(weights, cfg, hyp, nc, resume, device, mode="default"):
    try:
        return load_oneflow_pretrained(weights, cfg, hyp, nc, resume, device, mode)
    except :
        return load_torch_pretrained(weights, cfg, hyp, nc, resume, device, mode)


def load_oneflow_pretrained(weights, cfg, hyp, nc, resume, device, mode="default"):
    from utils.general import LOGGER

    if mode == "default":
        from models.yolo import Model as Model
    elif mode == "seg":
        from models.yolo import SegmentationModel as Model
    elif mode == "cls":
        pass
    else:
        assert mode in ["default", "seg", "cls"]

    ckpt = flow.load(
        weights, map_location="cpu"
    )  # load checkpoint to CPU to avoid CUDA memory leak
    model = Model(
        cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")
    ).to(device)
    exclude = (
        ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []
    )  # exclude keys

    csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  # load

    LOGGER.info(
        f"load_oneflow_pretrained Transferred {len(csd)}/{len(model.state_dict())} items from {weights}"
    )  # report
    return model


def copy_model_attributes(b, a):
    import oneflow as torch
    # add attributes
    # Copy model attributes from b to a
    attributes = [
        x
        for x in dir(b)
        if not callable(getattr(b, x)) and not x.startswith("__") and not x[0] == "_"
    ]
    for attr in attributes:
        get_attr = getattr(b, attr)
        if torch.is_tensor(get_attr):
            get_attr = flow.tensor(get_attr.numpy())
        setattr(a, attr, getattr(b, attr))


def load_torch_pretrained(weights, cfg, hyp, nc, resume, device, mode="default"):
    """mode:选择模型类型
    """
    from utils.general import LOGGER
    import oneflow as torch

    if mode == "default":
        from models.yolo import Model as Model
    elif mode == "seg":
        from models.yolo import SegmentationModel as Model
    elif mode == "cls":
        pass
    else:
        print(f"{mode=} worr")
        raise ImportError

    ckpt = torch.load(
        weights, map_location="cpu"
    )  # load checkpoint to CPU to avoid CUDA memory leak
    model = Model(
        cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")
    ).to(device)

    csd = dict()
    for key, value in ckpt["model"].state_dict().items():
        if value.detach().cpu().numpy().dtype == np.float16:
            tval = flow.tensor(value.detach().cpu().numpy().astype(np.float32))
        else:
            tval = flow.tensor(value.detach().cpu().numpy())
        csd[key] = tval

    exclude = (
        ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []
    )  # exclude keys
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  # load
    copy_model_attributes(ckpt["model"], model)
    LOGGER.info(
        f"load_torch_pretrained Transferred {len(csd)}/{len(model.state_dict())} items from {weights}"
    )
    return model


def attempt_load_torch(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    import oneflow as torch
    import oneflow.nn as nn
    from models.yolo import Detect, Model
    from models.experimental import Ensemble
    from models.yolo import ClassificationModel
    from utils.torch_utils import model_info

    model = Ensemble()
    ckpt = torch.load(weights, map_location=device)  # load
    of_model = Model(cfg="/home/fengwen/one-yolov5/models/yolov5s.yaml")
    of_model = ClassificationModel(model=of_model,cutoff=10) 
    csd = dict()
    for key, value in ckpt["model"].state_dict().items():
        if value.detach().cpu().numpy().dtype == np.float16:
            tval = flow.tensor(value.detach().cpu().numpy().astype(np.float32))
        else:
            tval = flow.tensor(value.detach().cpu().numpy())
        csd[key] = tval

    of_model.load_state_dict(csd, strict=False)  # load
    copy_model_attributes(ckpt["model"], of_model)  # add attributes
    # Model compatibility updates
    if not hasattr(of_model, "stride"):
        of_model.stride = flow.tensor([32.0])
    if hasattr(of_model, "names") and isinstance(of_model.names, (list, tuple)):
        of_model.names = dict(enumerate(of_model.names))  # convert to dict

    model.append(
        of_model.fuse().eval()
        if fuse and hasattr(of_model, "fuse")
        else of_model.eval()
    )  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [flow.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[
        flow.argmax(flow.tensor([m.stride.max() for m in model])).int()
    ].stride  # max stride
    assert all(
        model[0].nc == m.nc for m in model
    ), f"Models have different class counts: {[m.nc for m in model]}"
    return model

