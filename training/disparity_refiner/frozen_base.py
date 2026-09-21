"""Opt-in SSR-only training checks; no changes to legacy detach/joint behavior."""
from __future__ import annotations

import hashlib
from functools import wraps
import torch


def set_training_mode(model):
    if getattr(model, "refiner_only", False):
        model.eval()
        model.disparity_refiner.train()
    else:
        model.train()


def preserve_training_mode(function):
    @wraps(function)
    def wrapped(model, *args, **kwargs):
        flags = [(module, module.training) for module in model.modules()]
        try:
            return function(model, *args, **kwargs)
        finally:
            for module, flag in flags:
                module.training = flag
    return wrapped


def base_sha256(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not name.startswith("disparity_refiner."):
            value = value.detach().cpu().contiguous()
            digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_frozen_base(model):
    if not getattr(model, "refiner_only", False):
        return
    if any(p.requires_grad or p.grad is not None for name, p in model.named_parameters()
           if not name.startswith("disparity_refiner.")):
        raise RuntimeError("Frozen Base has trainable parameters or gradients")
    if base_sha256(model) != model.frozen_base_sha256:
        raise RuntimeError("Frozen Base parameters/buffers changed")
