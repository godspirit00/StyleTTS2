"""Helpers shared by the StyleTTS2 training scripts.

Everything here is used by train_first.py, train_second.py and
train_finetune.py (train_finetune_accelerate.py runs train_finetune.py), so
dynamic batching, mixed precision, gradient clipping and OOM recovery behave
identically in every stage.
"""
import gc
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

HOP = 300


# ---------------------------------------------------------------------------
# Runtime / speed setup
# ---------------------------------------------------------------------------

def resolve_device(config):
    dev = str(config.get('device', 'cuda'))
    if dev.startswith('cuda') and not torch.cuda.is_available():
        logger.warning('CUDA is not available; training on the CPU')
        return 'cpu'
    return dev


def setup_speed_flags(config):
    """TF32 matmuls/convs (Ampere+) and optional cuDNN autotuning."""
    if not torch.cuda.is_available():
        return
    tf32 = bool(config.get('allow_tf32', True))
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    # cuDNN autotuning pays off with dynamic batching (a bounded set of
    # shapes) but thrashes with the legacy random-length batches.
    torch.backends.cudnn.benchmark = bool(config.get('cudnn_benchmark', False))


def is_oom(e):
    msg = str(e).lower()
    return isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in msg or \
        ('cufft' in msg and 'alloc' in msg)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Mixed precision + optimizer stepping
# ---------------------------------------------------------------------------

def _parse_precision(spec):
    if spec is None or spec is False:
        return None
    if spec is True:
        return torch.float16
    s = str(spec).strip().lower()
    if s in ('no', 'none', 'off', 'false', '', 'fp32', 'float32'):
        return None
    if s in ('bf16', 'bfloat16'):
        return torch.bfloat16
    if s in ('fp16', 'float16', 'half'):
        return torch.float16
    raise ValueError('Unrecognized mixed_precision value: %r' % (spec,))


class Amp:
    """Autocast + (for fp16) a GradScaler that is correct for StyleTTS2's
    several backward passes and per-module optimizers per iteration.

    Usage per backward pass::

        amp.backward(loss)
        amp.step(optimizer, ['decoder', 'text_encoder'])   # unscale, clip, step
        amp.update()                                        # once per pass

    ``step`` also clips each module's gradient norm to ``clip_norm`` (if set)
    and skips the update of a module whose gradients are not finite, so a
    single bad batch cannot poison the weights with NaN/Inf.
    """

    def __init__(self, precision='no', device='cuda', clip_norm=None):
        self.dtype = _parse_precision(precision)
        self.device_type = 'cuda' if str(device).startswith('cuda') else 'cpu'
        if self.dtype is torch.bfloat16 and self.device_type == 'cuda' and \
                not torch.cuda.is_bf16_supported():
            logger.warning('bf16 is not supported on this GPU; falling back to fp16')
            self.dtype = torch.float16
        if self.dtype is torch.float16 and self.device_type != 'cuda':
            logger.warning('fp16 autocast needs CUDA; disabling mixed precision')
            self.dtype = None
        self.enabled = self.dtype is not None
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.dtype is torch.float16))
        self.clip_norm = float(clip_norm) if clip_norm else None
        self._unscaled = set()
        self.skipped_steps = 0

    def autocast(self):
        return torch.autocast(device_type=self.device_type, dtype=self.dtype or torch.float32,
                              enabled=self.enabled)

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def unscale(self, optimizer, keys):
        """Unscale the gradients of ``keys`` (needed before reading grads)."""
        if not self.scaler.is_enabled():
            return
        for key in keys:
            opt = optimizer.optimizers[key]
            if id(opt) in self._unscaled or not opt.param_groups:
                continue
            self.scaler.unscale_(opt)
            self._unscaled.add(id(opt))

    def step(self, optimizer, keys):
        self.unscale(optimizer, keys)
        for key in keys:
            opt = optimizer.optimizers[key]
            if not opt.param_groups:
                continue  # frozen module (no-op optimizer)
            params = [p for g in opt.param_groups for p in g['params'] if p.grad is not None]
            if not params:
                continue
            if self.clip_norm is not None:
                norm = torch.nn.utils.clip_grad_norm_(params, self.clip_norm)
            else:
                norm = None
            if self.scaler.is_enabled():
                self.scaler.step(opt)  # skips the update itself on inf/nan
            else:
                if norm is None:
                    norm = grad_norm(params)
                if not torch.isfinite(norm):
                    self.skipped_steps += 1
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.step()

    def update(self):
        self.scaler.update()
        self._unscaled.clear()

    def state_dict(self):
        return self.scaler.state_dict()

    def load_state_dict(self, state):
        if state:
            self.scaler.load_state_dict(state)


def grad_norm(params):
    grads = [p.grad.detach().float() for p in params if p.grad is not None]
    if not grads:
        return torch.zeros(())
    return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))


def module_grad_norm(module):
    return float(grad_norm([p for p in module.parameters() if p.requires_grad]))


def all_finite(*losses):
    for l in losses:
        if isinstance(l, torch.Tensor) and not torch.isfinite(l).all():
            return False
    return True


# ---------------------------------------------------------------------------
# Learning-rate schedule driven by training progress
# ---------------------------------------------------------------------------

class ProgressLR:
    """Warmup + cosine decay computed from the fraction of training done.

    A step-count schedule (OneCycleLR) breaks with dynamic batching, because
    the number of steps per epoch changes whenever batch sizes shrink or grow.
    Here the LR only depends on ``epoch + i / steps_in_epoch``.

    ``kind='constant'`` leaves the optimizer LRs untouched (the original
    StyleTTS2 behaviour: its OneCycleLR scheduler was never stepped).
    """

    def __init__(self, optimizer, total_epochs, kind='constant', warmup=0.1,
                 start_factor=0.1, final_factor=0.001):
        self.optimizer = optimizer
        self.total = max(1e-8, float(total_epochs))
        self.kind = kind
        self.warmup = float(warmup)
        self.start_factor = float(start_factor)
        self.final_factor = float(final_factor)
        self.base = {k: [g['lr'] for g in opt.param_groups]
                     for k, opt in optimizer.optimizers.items()}

    def factor(self, progress):
        p = min(max(progress / self.total, 0.0), 1.0)
        if self.kind == 'constant':
            return 1.0
        if p < self.warmup:
            return self.start_factor + (1 - self.start_factor) * p / self.warmup
        q = (p - self.warmup) / max(1e-8, 1 - self.warmup)
        return self.final_factor + (1 - self.final_factor) * 0.5 * (1 + math.cos(math.pi * q))

    def set_progress(self, progress):
        if self.kind == 'constant':
            return
        f = self.factor(progress)
        for k, opt in self.optimizer.optimizers.items():
            for g, base in zip(opt.param_groups, self.base[k]):
                g['lr'] = base * f


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def is_uniform(mel_input_length):
    return bool((mel_input_length == mel_input_length[0]).all())


def global_styles(model, mels, mel_input_length, need_grad_prosody=True):
    """Global acoustic (style_encoder) and prosodic (predictor_encoder) styles
    of the whole utterances.

    With dynamic batching all utterances have the same length and this is a
    single batched call; otherwise each utterance is encoded on its own (the
    encoders' average pooling does not support masking).  The acoustic style
    is only used as a detached diffusion target, so it is computed without
    building an autograd graph."""
    if is_uniform(mel_input_length):
        x = mels.unsqueeze(1)
        with torch.set_grad_enabled(need_grad_prosody and torch.is_grad_enabled()):
            s_dur = model.predictor_encoder(x)
        with torch.no_grad():
            gs = model.style_encoder(x)
        return s_dur, gs
    ss, gss = [], []
    for bib in range(len(mel_input_length)):
        mel = mels[bib, :, :mel_input_length[bib]].unsqueeze(0).unsqueeze(1)
        with torch.set_grad_enabled(need_grad_prosody and torch.is_grad_enabled()):
            ss.append(model.predictor_encoder(mel))
        with torch.no_grad():
            gss.append(model.style_encoder(mel))
    return torch.cat(ss, 0), torch.cat(gss, 0)


def duration_losses(d, d_gt, input_lengths):
    """Duration CE (per-frame binary targets) + L1 on summed durations.

    Vectorised version of the original per-phoneme Python loop (which issued
    one GPU sync per phoneme)."""
    B, T, max_dur = d.shape
    lengths = input_lengths.to(d.device)
    steps = torch.arange(max_dur, device=d.device)
    trg = (steps.view(1, 1, -1) < d_gt.long().unsqueeze(-1)).to(d.dtype)
    tok = torch.arange(T, device=d.device).view(1, -1)
    valid = tok < lengths.view(-1, 1)                                  # [B, T]
    inner = valid & (tok >= 1) & (tok < (lengths - 1).view(-1, 1))     # skip pads
    ce = F.binary_cross_entropy_with_logits(d.float(), trg.float(), reduction='none').mean(-1)
    loss_ce = ((ce * valid).sum(-1) / valid.sum(-1).clamp(min=1)).mean()
    dur_pred = torch.sigmoid(d.float()).sum(-1)
    l1 = (dur_pred - d_gt.float()).abs()
    loss_dur = ((l1 * inner).sum(-1) / inner.sum(-1).clamp(min=1)).mean()
    return loss_ce, loss_dur


def s2s_loss(s2s_pred, texts, input_lengths):
    """Per-utterance token CE of the text aligner (TMA), batch-averaged."""
    s2s_pred = s2s_pred[:, :texts.shape[1]]  # the decoder emits one extra step
    tok = torch.arange(texts.shape[1], device=texts.device).view(1, -1)
    valid = tok < input_lengths.to(texts.device).view(-1, 1)
    ce = F.cross_entropy(s2s_pred.float().transpose(1, 2), texts, reduction='none')
    return ((ce * valid).sum(-1) / valid.sum(-1).clamp(min=1)).mean()


def select_segments(asr, mels, waves, mel_input_length, max_len, device,
                    p=None, full_length=False, st_mode='same', min_style_frames=80):
    """Select the training segments for the decoder.

    ``full_length`` (dynamic batching): every item of the batch has the same
    padded length, so the *whole* utterance is used - no cropping at all,
    except that it is capped at ``max_len`` frames if ``max_len`` is set.

    Otherwise (legacy random batches) this is the original behaviour: every
    item is randomly cropped to the length of the shortest one in the batch
    (capped at ``max_len``).

    Returns ``en, gt, wav, st, p_en`` (``p_en`` is None when ``p`` is None).
    ``st_mode``: 'same' (style reference = gt) or 'random' (a different random
    crop of the same utterance, used by multi-speaker models so the style
    encoder cannot simply copy the target's content).
    """
    B = mels.shape[0]
    L = int(mel_input_length.min().item())
    if full_length and is_uniform(mel_input_length):
        half = L // 2
        if max_len:
            half = min(half, int(max_len) // 2)
    else:
        half = L // 2 - 1
        if max_len:
            half = min(half, int(max_len) // 2)
    # never read past the end of the shortest waveform (legacy padding makes
    # waveforms slightly shorter than mel_frames * hop)
    half = min(half, min(len(w) for w in waves) // (2 * HOP))
    half = max(half, 1)
    if full_length:
        # a random crop of about half the utterance
        st_half = min(L // 2, max(min_style_frames // 2, L // 4))
    else:
        st_half = L // 2 - 1  # original: shortest length in the batch

    en, gt, wav, st, p_en = [], [], [], [], []
    for bib in range(B):
        mel_length = min(int(mel_input_length[bib].item()) // 2,
                         len(waves[bib]) // (2 * HOP))
        start = np.random.randint(0, mel_length - half + 1) if mel_length > half else 0
        en.append(asr[bib, :, start:start + half])
        if p is not None:
            p_en.append(p[bib, :, start:start + half])
        gt.append(mels[bib, :, start * 2:(start + half) * 2])
        y = waves[bib][start * 2 * HOP:(start + half) * 2 * HOP]
        wav.append(torch.from_numpy(np.ascontiguousarray(y)).float())
        if st_mode == 'random':
            s0 = np.random.randint(0, mel_length - st_half + 1) if mel_length > st_half else 0
            st.append(mels[bib, :, s0 * 2:(s0 + st_half) * 2])
    en = torch.stack(en)
    gt = torch.stack(gt).detach()
    wav = torch.stack(wav).to(device, non_blocking=True).detach()
    st = torch.stack(st).detach() if st else gt
    p_en = torch.stack(p_en) if p is not None else None
    return en, gt, wav, st, p_en


def safe_mean(values):
    return float(np.mean(values)) if len(values) else float('nan')


def resolve_precision(config):
    """``mixed_precision`` from the config; when it is 'no', honour
    ``accelerate launch --mixed_precision=...`` (ACCELERATE_MIXED_PRECISION)."""
    import os
    mp = config.get('mixed_precision', 'no')
    if isinstance(mp, bool):
        mp = 'fp16' if mp else 'no'
    if _parse_precision(mp) is None:
        mp = os.environ.get('ACCELERATE_MIXED_PRECISION', 'no')
    return mp


def phase_scale(config, phase):
    """Initial batch-size scale of a dynamic-batching phase the first time it
    is entered (``phase_batch_scale: {phase: factor}`` in the config)."""
    return float((config.get('phase_batch_scale') or {}).get(phase, 1.0))


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)
