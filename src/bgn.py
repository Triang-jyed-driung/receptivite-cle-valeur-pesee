import torch
class BackGradNorm(torch.nn.Module):
    def __init__(self, dim=(0,1,2), eps=1e-24):
        super().__init__()
        self.dim = dim
        self.eps = eps
    def forward(self, x):
        return x
    def backward(self, dx):
        return torch.nn.functional.normalize(dx, dim=(0,1,2), eps=self.eps)