# One-YOLOv5 🚀 fix by OneFlow
import os
import psutil
import subprocess
import threading
import time
import threading
import oneflow as flow
import numpy as np


class FlowCudaMemoryReserved:
    """
    This function returns the memory usage of the current process.
    Parameters
    ----------
    pid : int
        The process ID of the process to get the memory usage of.
    Returns
    -------
    memory : float
        The memory usage of the process in MB.

    """

    def __init__(self, device_type="GPU", default_update_time=30.0) -> None:
        self._device_type = device_type
        self._pid = str(os.getpid())
        self._current_mem = None
        self._update_time = default_update_time
        self._threshold = 30 * 60  # 30 min
        thread = threading.Thread(target=self.update_data, args=())
        thread.daemon = True
        thread.start()
        print(f"current pid {self._pid=}")

    def adjust_interval(self, previous_mem, current_mem):
        try:
            if self._update_time > self._threshold:
                return
            if current_mem == "None" or previous_mem == None or previous_mem == "None":
                return
            # If the difference between current_mem and previous_mem is small, increase the _update_time
            if abs(float(current_mem) - float(previous_mem)) < 0.1:
                self._update_time *= 1.5
        except:
            pass

    def __call__(self, mode="MB"):
        """
        This function returns the memory usage of the current process.
        Parameters:
            mode: "MB" or "GB"
        Returns:
            memory usage of the current process.
        """
        memory = self._current_mem
        try:
            if mode == "MB":
                return "%.3fMB" % (float(memory))
            elif mode == "GB":
                return "%.3fG" % (float(memory) / 1024)
            else:
                return "None"
        except:
            return "None"

    def update_data(self):
        while True:
            previous_mem = self._current_mem
            self._current_mem = self.use_memory(mode="MB")
            self.adjust_interval(previous_mem=previous_mem, current_mem=self._current_mem)
            time.sleep(self._update_time)

    def use_memory(self, mode="MB"):
        pid = self._pid
        try:
            memory = self.get_gpu_memory(pid) if self._device_type == "GPU" else self.get_cpu_memory(pid)
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
        process_memory = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"])
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
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def load_pretrained(weights, cfg, hyp, nc, resume, device, mode="default"):
    """
    Load a PyTorch model from a checkpoint.
    Parameters
    ----------
    weights : str
        Path to weights file.
    cfg : str
        Path to model configuration file.
    hyp : dict
        Hyperparameters.
    nc : int
        Number of classes.
    resume : bool
        Whether to resume training.
    device : torch.device
        Device to load weights on.
    mode : str
        One of 'default', 'seg', or 'cls'.

    Returns
    -------
    model : torch.nn.Module
        Model loaded from weights.
    """
    try:
        return load_oneflow_pretrained(weights, cfg, hyp, nc, resume, device, mode)
    except:
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

    ckpt = flow.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
    model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)
    exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # exclude keys

    csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  # load

    LOGGER.info(f"load_oneflow_pretrained Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
    return ckpt, csd, model


def copy_model_attributes(b, a):
    import oneflow as torch

    # add attributes
    # Copy model attributes from b to a
    attributes = [x for x in dir(b) if not callable(getattr(b, x)) and not x.startswith("__") and not x[0] == "_"]
    for attr in attributes:
        get_attr = getattr(b, attr)
        if torch.is_tensor(get_attr):
            get_attr = flow.tensor(get_attr.numpy())
        setattr(a, attr, getattr(b, attr))


def load_torch_pretrained(weights, cfg, hyp, nc, resume, device, mode="default"):
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

    ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
    model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)

    csd = dict()
    for key, value in ckpt["model"].state_dict().items():
        if value.detach().cpu().numpy().dtype == np.float16:
            tval = flow.tensor(value.detach().cpu().numpy().astype(np.float32))
        else:
            tval = flow.tensor(value.detach().cpu().numpy())
        csd[key] = tval

    exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # exclude keys
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  # load
    copy_model_attributes(ckpt["model"], model)
    LOGGER.info(f"load_torch_pretrained Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")
    return ckpt, csd, model
