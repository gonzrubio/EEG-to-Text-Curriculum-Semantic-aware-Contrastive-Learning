"""Wrapper to store tensors in a set.

Note: CPU support only.
"""

import torch


class HashTensor:
    def __init__(self, obj):
        self.obj = obj

    def __hash__(self):
        return hash(self.obj.cpu().numpy().tobytes())

    def __eq__(self, other):
        if isinstance(other, HashTensor):
            return torch.equal(self.obj, other.obj)
        elif isinstance(other, torch.Tensor):
            return torch.equal(self.obj, other)
        return False

    def __repr__(self):
        return repr(self.obj)

    def __getitem__(self, index):
        return self.obj[index]

    def __getattr__(self, attr):
        if hasattr(self.obj, attr):
            return getattr(self.obj, attr)
        else:
            raise AttributeError(
                f"'{type(self.obj).__name__}' object has no attribute '{attr}'"
                )
