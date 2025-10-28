import torch
from torchvision.utils import save_image

EUROSAT_MEAN = [0.3444, 0.3803, 0.4078]
EUROSAT_STD = [0.0914, 0.0651, 0.0552]

def unnormalize_to01(x: torch.Tensor,
                     mean=EUROSAT_MEAN,
                     std=EUROSAT_STD) -> torch.Tensor:
    """
    x: (B,C,H,W) in normalized space (what the model sees).
    returns: unnormalized to [0,1] for saving.
    """
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    s = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    x01 = x * s + m
    return x01.clamp(0.0, 1.0)
