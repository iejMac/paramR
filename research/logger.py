import os
import numpy as np

class BinaryLogger:
    def __init__(self, log_dir, n_steps, metrics):
        self.log_dir = log_dir
        self.n_steps = n_steps

        self.metrics = {}
        for m_name, m_shape in metrics.items():
            m_full_shape = (n_steps,) + m_shape
            self.metrics[m_name] = np.zeros(m_full_shape, dtype=np.float64)
   
    def log(self, metric):
        step = metric.pop("step")
        for m_name, m in metric.items():
            self.metrics[m_name][step] = m

    def save(self):
        for m_name, m_dat in self.metrics.items():
            np.save(os.path.join(self.log_dir, f"{m_name}.npy"), m_dat)
