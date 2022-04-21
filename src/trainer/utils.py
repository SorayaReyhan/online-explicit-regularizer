from typing import Dict

import torch


def normalize(saliency_map: Dict[str, torch.Tensor]):
    total = 0
    for saliency in saliency_map.values():
        total += saliency.sum()
    saliency_map = {name: saliency / total for name, saliency in saliency_map.items()}
    return saliency_map
