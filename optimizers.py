#coding:utf-8
import os, sys
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from functools import reduce
from torch.optim import AdamW


# Lazy import for bitsandbytes 8-bit AdamW so the dependency stays optional.
def _try_get_adamw8bit():
    try:
        import bitsandbytes as bnb  # type: ignore
        return bnb.optim.AdamW8bit
    except Exception:
        return None


def _build_adamw(params, lr, *, use_8bit=False, fused=True, eps=1e-9,
                 weight_decay=1e-4, betas=(0.0, 0.99)):
    """Build an AdamW optimizer.  Falls back gracefully when 8-bit is requested
    but bitsandbytes is unavailable, or when fused kernels are unsupported.
    """
    params = list(params)
    if use_8bit:
        AdamW8bit = _try_get_adamw8bit()
        if AdamW8bit is not None:
            return AdamW8bit(params, lr=lr, weight_decay=weight_decay,
                             betas=betas, eps=eps)
        # bnb unavailable: fall through to fused/regular AdamW.

    if fused and torch.cuda.is_available():
        try:
            return AdamW(params, lr=lr, weight_decay=weight_decay,
                         betas=betas, eps=eps, fused=True)
        except (TypeError, RuntimeError):
            # Older PyTorch / unsupported device: fall back to non-fused.
            pass
    return AdamW(params, lr=lr, weight_decay=weight_decay,
                 betas=betas, eps=eps)


class MultiOptimizer:
    def __init__(self, optimizers={}, schedulers={}):
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.keys = list(optimizers.keys())
        self.param_groups = reduce(lambda x,y: x+y, [v.param_groups for v in self.optimizers.values()])

    def state_dict(self):
        state_dicts = [(key, self.optimizers[key].state_dict())\
                       for key in self.keys]
        return state_dicts

    def load_state_dict(self, state_dict):
        for key, val in state_dict:
            try:
                self.optimizers[key].load_state_dict(val)
            except:
                print("Unloaded %s" % key)

    def step(self, key=None, scaler=None):
        keys = [key] if key is not None else self.keys
        _ = [self._step(key, scaler) for key in keys]

    def _step(self, key, scaler=None):
        if scaler is not None:
            scaler.step(self.optimizers[key])
            scaler.update()
        else:
            self.optimizers[key].step()

    def zero_grad(self, key=None):
        if key is not None:
            self.optimizers[key].zero_grad()
        else:
            _ = [self.optimizers[key].zero_grad() for key in self.keys]

    def scheduler(self, *args, key=None):
        if key is not None:
            self.schedulers[key].step(*args)
        else:
            _ = [self.schedulers[key].step(*args) for key in self.keys]

def define_scheduler(optimizer, params):
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params.get('max_lr', 2e-4),
        epochs=params.get('epochs', 200),
        steps_per_epoch=params.get('steps_per_epoch', 1000),
        pct_start=params.get('pct_start', 0.0),
        div_factor=params.get('div_factor', 1),
        final_div_factor=params.get('final_div_factor', 1))

    return scheduler

def build_optimizer(parameters_dict, scheduler_params_dict, lr,
                    use_8bit=False, fused=True):
    """Build the per-module AdamW optimizers used by StyleTTS2.

    Modules whose parameters are all frozen (``requires_grad=False``) get a
    no-op optimizer instead of one allocating Adam state for unused params.

    Args:
        use_8bit: if True, use bitsandbytes.optim.AdamW8bit when available
            (~4x smaller optimizer state).  Falls back to standard AdamW.
        fused:   if True, use the fused-AdamW CUDA kernel when available
            (a small speedup).  Falls back to non-fused on older PyTorch.
    """
    optim = {}
    for key, params in parameters_dict.items():
        params = [p for p in params if p.requires_grad]
        if len(params) == 0:
            # Frozen module: provide a no-op optimizer so step()/zero_grad()
            # still work without allocating Adam state for unused tensors.
            optim[key] = _NoOpOptimizer()
        else:
            optim[key] = _build_adamw(
                params, lr=lr, use_8bit=use_8bit, fused=fused,
                weight_decay=1e-4, betas=(0.0, 0.99), eps=1e-9,
            )

    schedulers = {}
    for key, opt in optim.items():
        if isinstance(opt, _NoOpOptimizer):
            schedulers[key] = _NoOpScheduler()
        else:
            schedulers[key] = define_scheduler(opt, scheduler_params_dict[key])

    return MultiOptimizer(optim, schedulers)


class _NoOpOptimizer:
    """Stub that satisfies the MultiOptimizer API for fully-frozen modules."""

    def __init__(self):
        self.param_groups = []
        self.state = {}

    def step(self, *args, **kwargs):
        pass

    def zero_grad(self, *args, **kwargs):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class _NoOpScheduler:
    def step(self, *args, **kwargs):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
