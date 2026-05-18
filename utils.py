from monotonic_align import maximum_path
from monotonic_align import mask_from_lens
from monotonic_align.core import maximum_path_c
import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
from munch import Munch

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent =  np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
  path =  np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

  t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
  t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)


# ---------------------------------------------------------------------------
# Training helpers: mixed precision + gradient checkpointing
# ---------------------------------------------------------------------------

def resolve_amp_dtype(spec):
    """Map a config 'mixed_precision' value to (autocast_enabled, dtype).

    Accepted spec values:
        - "no" / False / None / ""   -> AMP disabled
        - "bf16" / "bfloat16"        -> autocast(bf16) (Ampere+ / cuda 8.0+)

    fp16 is rejected because the manual-AMP scripts (train_second.py,
    train_finetune.py) issue multiple ``backward()`` calls per iteration and
    the GradScaler integration required to keep fp16 stable is not wired in.
    Use ``bf16`` on Ampere+ GPUs, or use the Accelerate-based scripts
    (train_first.py / train_finetune_accelerate.py) which scale internally.
    """
    if spec is None or spec is False:
        return False, torch.float32
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in ("no", "none", "off", "false", ""):
            return False, torch.float32
        if s in ("bf16", "bfloat16"):
            return True, torch.bfloat16
        if s in ("fp16", "float16", "half"):
            raise ValueError(
                "mixed_precision='fp16' is not supported in the manual-AMP "
                "training scripts because GradScaler is not wired into the "
                "multi-backward loop and unscaled fp16 grads will overflow. "
                "Use 'bf16' instead, or switch to the Accelerate-based "
                "scripts (train_first.py / train_finetune_accelerate.py).")
    raise ValueError(f"Unrecognized mixed_precision value: {spec!r}")


def enable_diffusion_gradient_checkpointing(enabled: bool) -> None:
    """Toggle gradient checkpointing inside the diffusion transformer blocks.

    Cuts activation memory at the cost of one extra forward pass per block
    during backward.  Safe to call before training starts.
    """
    try:
        from Modules.diffusion.modules import set_gradient_checkpointing
        set_gradient_checkpointing(enabled)
    except Exception:
        pass


def freeze_module(module):
    """Disable gradient tracking and switch to eval-mode for ``module``."""
    for p in module.parameters():
        p.requires_grad = False
    module.eval()
    return module


def configure_cuda_allocator(expandable_segments: bool = True,
                             max_split_size_mb=None) -> None:
    """Configure the PyTorch CUDA caching allocator.

    Must run before any CUDA tensor is created (i.e. at the very top of
    training, before ``.to('cuda')``).  Sets ``PYTORCH_CUDA_ALLOC_CONF``
    if it has not already been set in the environment.

    ``expandable_segments=True`` (PyTorch 2.1+) is the standard fix for
    fragmentation in workloads with variable-shape allocations such as
    StyleTTS2's dynamic batching.
    """
    import os
    if 'PYTORCH_CUDA_ALLOC_CONF' in os.environ:
        # Respect any user-supplied configuration.
        return
    parts = []
    if expandable_segments:
        parts.append('expandable_segments:True')
    if max_split_size_mb is not None:
        parts.append(f'max_split_size_mb:{int(max_split_size_mb)}')
    if parts:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ','.join(parts)


def cuda_memory_summary() -> str:
    """Return a short string with current/peak CUDA memory usage (in GiB)."""
    if not torch.cuda.is_available():
        return 'CUDA unavailable'
    gib = 1024 ** 3
    return (
        'cuda mem: '
        f'allocated={torch.cuda.memory_allocated() / gib:.2f} GiB, '
        f'reserved={torch.cuda.memory_reserved() / gib:.2f} GiB, '
        f'peak_allocated={torch.cuda.max_memory_allocated() / gib:.2f} GiB, '
        f'peak_reserved={torch.cuda.max_memory_reserved() / gib:.2f} GiB'
    )


def recover_from_oom(local_vars_to_drop, optimizer=None, logger=None) -> None:
    """Free as much VRAM as possible after a CUDA OOM.

    The default OOM handler just calls ``empty_cache``; the autograd graph
    attached to the iteration's local tensors stays alive until the next
    iteration overwrites them, which keeps a large chunk of VRAM tied up
    and often triggers a second OOM.  This helper drops those references
    explicitly, runs gc, and resets the allocator.

    Args:
        local_vars_to_drop: a list of names that the caller wants cleared.
            Pass the calling frame's ``locals()`` if you want all locals
            scrubbed.
        optimizer: optional ``MultiOptimizer`` — its grads are zeroed so the
            backward-graph references are released too.
        logger: optional logger for diagnostic output.
    """
    import gc
    if optimizer is not None:
        try:
            optimizer.zero_grad()
        except Exception:
            pass
    # Drop references to the iteration's locals so their graphs are freed.
    if isinstance(local_vars_to_drop, dict):
        for k in list(local_vars_to_drop.keys()):
            local_vars_to_drop[k] = None
    elif isinstance(local_vars_to_drop, list):
        for i in range(len(local_vars_to_drop)):
            local_vars_to_drop[i] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if logger is not None:
            try:
                logger.info('Post-OOM ' + cuda_memory_summary())
            except Exception:
                pass
