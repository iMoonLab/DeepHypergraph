import numpy as np
import torch


class CIndexMeter:
    def __init__(self):
        super(CIndexMeter, self).__init__()
        self.reset()

    def reset(self):
        self.output = np.array([])
        self.target = np.array([])

    def add(self, output: torch.tensor, target: torch.tensor):
        output = output.cpu().detach().squeeze().numpy()[np.newaxis]
        target = target.cpu().detach().squeeze().numpy()[np.newaxis]

        assert output.ndim == target.ndim, 'target and output do not match'
        assert output.ndim == 1

        self.output = np.hstack([self.output, output])
        self.target = np.hstack([self.target, target])

    def value(self):
        output = self.output[np.newaxis]
        target = self.target[np.newaxis]

        num_sample = output.shape[-1]
        num_hit = (~((output.T > output) ^ (target.T > target))).sum()

        return float(num_hit - num_sample) / float(num_sample * num_sample - num_sample)
