# One-YOLOv5 🚀 fix by OneFlow
import os  # noqa :E402
import psutil  # noqa :E402
import subprocess  # noqa :E402
import threading  # noqa :E402
import time  # noqa :E402
import threading  # noqa :E402
import oneflow as flow  # noqa :E402
import numpy as np  # noqa :E402


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
            if current_mem == "None" or previous_mem == "None":
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