import torch

class NeuronGroup(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.idx = None