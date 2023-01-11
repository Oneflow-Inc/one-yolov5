# One-YOLOv5 🚀 by OneFlow
"""
issues:
函数名:
函数的参数:
函数的返回值:
"""
"""
缺少：
1. https://pytorch.org/docs/stable/mobile_optimizer.html?highlight=optimize_for_mobile#torch.utils.mobile_optimizer.optimize_for_mobile
2. Tensor' object has no attribute 'gt_ 

"""
"""
未对齐:
1. i = flowvision.ops.nms(boxes, scores, iou_thres)  # NMS flow.nms()
2. imgs = imgs.to(device, non_blocking=True).float 
"""

import os
import shutil
import psutil
import subprocess
import threading
import time
import threading
import oneflow as flow


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


def get_file_size(path):
    # Print the size of all files in a directory and its subdirectories (in bytes).
    if not os.path.isdir(path):
        return os.path.getsize(path)

    size = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            file_path = os.path.join(root, f)
            size += os.path.getsize(file_path)
    return size


def model_save(obj, path_file) -> None:
    try:
        if os.path.exists(path_file):
            if os.path.isdir(path_file):
                shutil.rmtree(path_file)
            else:
                os.remove(path_file)
        flow.save(obj, path_file)
        return True
    except Exception:
        print(f"warning model save failed  in {path_file}❌")
        return False


def tensor_gt_(tensor: flow.tensor, other):
    """
    issues: https://github.com/Oneflow-Inc/oneflow/issues/9563
    """
    dtype = tensor.dtype
    result = tensor.gt(other)
    return flow.tensor(result.numpy(), dtype=dtype, device=tensor.device)

