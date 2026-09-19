"""NVML sampler thread shared by the Job C scripts.

Samples every NVML-visible GPU at a fixed period and writes
wall_time_s,nvml_index,uuid,temp_gpu_C,temp_mem_C,power_W,util_gpu_pct
to a CSV. Memory temperature uses field NVML_FI_DEV_MEMORY_TEMP when the
driver exposes it (blank otherwise).
"""
import threading
import time

import pynvml

NVML_FI_DEV_MEMORY_TEMP = 82


class Sampler(threading.Thread):
    def __init__(self, path, period_s=0.02):
        super().__init__(daemon=True)
        self.path = path
        self.period_s = period_s
        self._stop_evt = threading.Event()
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
        self.uuids = []
        for h in self.handles:
            u = pynvml.nvmlDeviceGetUUID(h)
            self.uuids.append(u.decode() if isinstance(u, bytes) else u)
        self.n_samples = 0
        self.mem_temp_ok = True

    def _mem_temp(self, h):
        if not self.mem_temp_ok:
            return ""
        try:
            v = pynvml.nvmlDeviceGetFieldValues(h, [NVML_FI_DEV_MEMORY_TEMP])[0]
            if v.nvmlReturn != 0:
                self.mem_temp_ok = False
                return ""
            return str(v.value.uiVal)
        except Exception:
            self.mem_temp_ok = False
            return ""

    def run(self):
        with open(self.path, "w", buffering=1 << 16) as f:
            f.write("wall_time_s,nvml_index,uuid,temp_gpu_C,temp_mem_C,power_W,util_gpu_pct\n")
            next_t = time.time()
            while not self._stop_evt.is_set():
                t = time.time()
                for i, h in enumerate(self.handles):
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                        pw = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                        ut = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    except pynvml.NVMLError:
                        continue
                    f.write(f"{t:.4f},{i},{self.uuids[i]},{temp},{self._mem_temp(h)},{pw:.2f},{ut}\n")
                self.n_samples += 1
                next_t += self.period_s
                delay = next_t - time.time()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.time()

    def stop(self):
        self._stop_evt.set()
        self.join()
