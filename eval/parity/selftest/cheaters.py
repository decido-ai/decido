"""Predictors that violate the CPU-only rule; the runner must flag them (exit code 2)."""
import numpy as np


class MpsCheater:
    def __init__(self):
        import torch
        self.x = torch.ones(1024, 1024, device="mps")  # explicit MPS use despite is_available() == False

    def __call__(self, request):
        from eval.parity.common import option_keys
        return {"answers": {q: (np.ones(len(option_keys(v))) / len(option_keys(v))).tolist()
                            for q, v in request["questions"].items()}}
