#coding: utf-8
import os
import os.path as osp
import random
import json
import logging

import numpy as np
import soundfile as sf
import librosa
import pandas as pd  # noqa: F401  (kept for backwards compatibility of imports)

import torch
import torchaudio
from torch.utils.data import DataLoader

# The symbol table MUST be identical to the one used at inference time
# (text_utils.py); keep a single source of truth.
from text_utils import symbols, dicts

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# PL-BERT has 512 position embeddings; texts are wrapped in two pad tokens.
MAX_TEXT_TOKENS = 510


class TextCleaner:
    """Maps a phoneme string to token ids, silently skipping unknown chars
    (they are reported once per character instead of once per sample)."""

    _reported = set()

    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts

    def __call__(self, text):
        indexes = []
        for char in text:
            idx = self.word_index_dictionary.get(char)
            if idx is None:
                if char not in TextCleaner._reported:
                    TextCleaner._reported.add(char)
                    logger.warning('TextCleaner: unknown symbol %r dropped (first seen in: %s)',
                                   char, text.strip()[:80])
                continue
            indexes.append(idx)
        return indexes


np.random.seed(1)
random.seed(1)
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
}

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


# ---------------------------------------------------------------------------
# Dynamic batching
# ---------------------------------------------------------------------------
# The original code builds random batches and then crops *every* sample to the
# length of the shortest one in the batch, so the larger the batch the shorter
# (and more random) the training segments become.  With dynamic batching every
# sample is assigned to a length bin, zero-padded (symmetrically, like the
# original 5000-sample padding) to the bin's fixed length, and batches are
# drawn from a single bin.  All items of a batch therefore have exactly the
# same length and the training scripts use the full padded utterance - no
# cross-sample cropping at all.  The batch size of each bin is derived from a
# frame budget (short clips -> big batches, long clips -> small batches).

_HOP_LENGTH = 300  # must match SPECT_PARAMS hop_length
_BIN_FRAMES = 20   # width of one bin in mel frames (0.25 s)


def get_time_bin(sample_count):
    """Map an audio sample count to a bin index (bin k: [20+20k, 40+20k) frames)."""
    frames = sample_count // _HOP_LENGTH
    if frames >= 20:
        return (frames - 20) // _BIN_FRAMES
    return 0


def get_frame_count(bin_id):
    """Padded mel-frame length of a bin: 60 + 20*bin.

    The bin's longest clip has < 40 + 20*bin frames, so every clip gets
    between ~10 and ~20 frames (0.125-0.25 s) of silence on each side,
    comparable to the original fixed 5000-sample padding."""
    return bin_id * _BIN_FRAMES + 60


def get_bin_from_mel_input_length(mel_input_length):
    """Recover the bin id from a dynamic batch's mel lengths (all equal)."""
    frame_count = int(mel_input_length.min().item())
    return max(0, (frame_count - 60) // _BIN_FRAMES)


# Bin 0 (60 frames) is shorter than the 80 frames the style encoder needs.
MIN_USABLE_BIN = 1


class BatchManager:
    """Decides (and learns) the batch size of every length bin.

    * Sizes come from a frame budget: ``max_batch_frames // bin_frames``
      (clamped to [1, max_batch_size]).
    * Memory use differs a lot between training phases (e.g. before/after the
      discriminators, diffusion or SLM adversarial training join), so sizes are
      tracked per *phase* (see :meth:`set_phase`); a phase can start from a
      scaled-down budget.
    * On OOM the bin's size is reduced (x0.75) and the failing size is
      remembered as a ceiling for that phase.  Bins that OOM repeatedly even at
      batch size 1 are skipped (0) for the phase, with a warning.
    * After ``grow_interval`` consecutive successful batches a bin whose size
      is below its budget size grows again (x1.125), never reaching a size that
      has OOMed, so an over-conservative start does not waste the GPU forever.

    Everything is persisted to ``json_path`` so a later run (or a resumed run)
    starts from the learned sizes.  Only the main process writes the file.
    """

    VERSION = 2

    def __init__(self, json_path, max_batch_frames, max_batch_size=64,
                 grow_interval=50, skip_after_ooms_at_1=3, is_main_process=True):
        self.json_path = json_path
        self.max_batch_frames = int(max_batch_frames)
        self.max_batch_size = int(max_batch_size)
        self.grow_interval = int(grow_interval)
        self.skip_after_ooms_at_1 = int(skip_after_ooms_at_1)
        self.is_main_process = is_main_process
        self.phases = {}
        self._load()
        self.phase = None
        self.set_phase('default')
        self._success = {}
        self._ooms_at_1 = {}

    # -- persistence --------------------------------------------------------
    def _load(self):
        if not self.json_path or not osp.exists(self.json_path):
            return
        try:
            with open(self.json_path) as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning('BatchManager: could not read %s (%s); starting fresh',
                           self.json_path, e)
            return
        if not isinstance(data, dict) or data.get('version') != self.VERSION:
            logger.warning('BatchManager: ignoring %s written by an older version '
                           '(delete it to silence this warning)', self.json_path)
            return
        for name, ph in data.get('phases', {}).items():
            if int(ph.get('budget', -1)) != self.max_batch_frames:
                # The user changed max_batch_frames: learned sizes are stale.
                continue
            self.phases[name] = {
                'budget': self.max_batch_frames,
                'scale': float(ph.get('scale', 1.0)),
                'sizes': {int(k): int(v) for k, v in ph.get('sizes', {}).items()},
                'ceil': {int(k): int(v) for k, v in ph.get('ceil', {}).items()},
            }

    def _save(self):
        if not self.is_main_process or not self.json_path:
            return
        data = {'version': self.VERSION, 'phases': {}}
        for name, ph in self.phases.items():
            data['phases'][name] = {
                'budget': ph['budget'], 'scale': ph['scale'],
                'sizes': {str(k): v for k, v in sorted(ph['sizes'].items())},
                'ceil': {str(k): v for k, v in sorted(ph['ceil'].items())},
            }
        tmp = self.json_path + '.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump(data, f, indent=1)
            os.replace(tmp, self.json_path)
        except OSError as e:
            logger.warning('BatchManager: could not write %s (%s)', self.json_path, e)

    # -- phases ---------------------------------------------------------------
    def set_phase(self, name, scale=1.0):
        """Switch to phase *name*.  A phase seen before (in this or an earlier
        run with the same budget) resumes its learned sizes; a new phase starts
        at ``scale`` x the budget sizes."""
        if name == self.phase:
            return
        if name not in self.phases:
            self.phases[name] = {'budget': self.max_batch_frames, 'scale': float(scale),
                                 'sizes': {}, 'ceil': {}}
        self.phase = name
        self._ph = self.phases[name]
        self._success = {}
        self._ooms_at_1 = {}
        self._save()

    # -- sizes ----------------------------------------------------------------
    def target(self, bin_id):
        """Budget-derived batch size for a bin (the growth target)."""
        bs = self.max_batch_frames // get_frame_count(bin_id)
        return max(1, min(self.max_batch_size, bs))

    def get(self, bin_id):
        sizes = self._ph['sizes']
        if bin_id in sizes:
            return sizes[bin_id]
        return max(1, int(self.target(bin_id) * self._ph['scale']))

    def _set(self, bin_id, size):
        self._ph['sizes'][bin_id] = int(size)
        self._save()

    def on_oom(self, bin_id, used_size):
        """Record an OOM for a batch of ``used_size`` items from ``bin_id``.
        Returns the new batch size of the bin (0 = bin skipped)."""
        cur = self.get(bin_id)
        ceil = self._ph['ceil']
        ceil[bin_id] = min(ceil.get(bin_id, used_size), used_size)
        self._success[bin_id] = 0
        if used_size > cur:
            # A batch that was prefetched before the last reduction: the size
            # has already been lowered, don't shrink again.
            self._save()
            return cur
        if cur <= 1:
            n = self._ooms_at_1.get(bin_id, 0) + 1
            self._ooms_at_1[bin_id] = n
            if n >= self.skip_after_ooms_at_1:
                logger.warning(
                    'BatchManager[%s]: bin %d (%d frames, %.2f s) runs out of memory '
                    'even with batch size 1; skipping it for this phase. Lower '
                    'max_len/max_segment_len, enable gradient_checkpointing or '
                    'mixed_precision, or split long utterances.', self.phase,
                    bin_id, get_frame_count(bin_id), get_frame_count(bin_id) * _HOP_LENGTH / 24000)
                self._set(bin_id, 0)
                return 0
            self._save()
            return 1
        new = min(cur - 1, max(1, int(cur * 0.75)))
        self._set(bin_id, new)
        logger.info('BatchManager[%s]: OOM in bin %d at batch size %d -> %d',
                    self.phase, bin_id, used_size, new)
        return new

    def on_success(self, bin_id, used_size):
        """Record a successful training step; may grow the bin's batch size."""
        cur = self.get(bin_id)
        if cur <= 0 or used_size < cur:
            return
        self._ooms_at_1.pop(bin_id, None)
        n = self._success.get(bin_id, 0) + 1
        if n < self.grow_interval:
            self._success[bin_id] = n
            return
        self._success[bin_id] = 0
        limit = min(self.target(bin_id), self._ph['ceil'].get(bin_id, 1 << 30) - 1)
        if cur < limit:
            new = min(limit, cur + max(1, cur // 8))
            self._set(bin_id, new)
            logger.info('BatchManager[%s]: bin %d batch size grown %d -> %d',
                        self.phase, bin_id, cur, new)

    # Backwards-compatible alias used by older scripts.
    def decrement(self, bin_id):
        before = self.get(bin_id)
        return self.on_oom(bin_id, before) < before


class DynamicBatchSampler(torch.utils.data.Sampler):
    """Yields batches whose items all come from the same length bin.

    Batches are generated lazily, reading the bin's batch size from the
    :class:`BatchManager` at the moment the batch is produced, so OOM
    reductions / growth take effect within the epoch (after the DataLoader's
    small prefetch queue drains).  Bins are interleaved randomly, weighted by
    the number of samples they still hold, so every part of the epoch sees a
    mix of lengths.

    For multi-process (DDP) training every rank uses the same seeded RNG and
    therefore produces the same sequence of global batches; each global batch
    holds ``batch_size * num_replicas`` items and rank ``r`` takes items
    ``r::num_replicas``.  Global batches that cannot give every rank at least
    one item are dropped so all ranks run the same number of steps.
    """

    def __init__(self, dataset, batch_manager, shuffle=True, num_replicas=1,
                 rank=0, seed=0):
        self.batch_manager = batch_manager
        self.shuffle = shuffle
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.seed = seed
        self.epoch = 0
        self.bins = {}
        skipped = 0
        for idx, bin_id in enumerate(dataset.sample_bins):
            if bin_id is None:
                continue  # unreadable / filtered sample
            if bin_id < MIN_USABLE_BIN:
                skipped += 1
                continue
            self.bins.setdefault(bin_id, []).append(idx)
        if skipped:
            logger.warning(
                'DynamicBatchSampler: excluded %d samples shorter than ~0.5 s '
                '(the style encoder needs >= 80 mel frames)', skipped)
        if not self.bins:
            raise ValueError(
                'DynamicBatchSampler: no usable training samples. Check '
                'data_params.root_path and the paths in your train list.')

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1  # a fresh order next time even if set_epoch is never called
        bins = {k: list(v) for k, v in sorted(self.bins.items())}
        if self.shuffle:
            for indices in bins.values():
                rng.shuffle(indices)
        cursors = {k: 0 for k in bins}
        world = self.num_replicas

        while True:
            live = [k for k in bins if cursors[k] < len(bins[k])]
            if not live:
                return
            if self.shuffle:
                weights = [len(bins[k]) - cursors[k] for k in live]
                bin_id = rng.choices(live, weights=weights)[0]
            else:
                bin_id = live[0]
            indices = bins[bin_id]
            c = cursors[bin_id]
            bs = self.batch_manager.get(bin_id)
            if bs <= 0:  # bin skipped for this phase
                cursors[bin_id] = len(indices)
                continue
            glob = indices[c:c + bs * world]
            cursors[bin_id] = c + len(glob)
            if len(glob) < world:
                continue
            yield glob[self.rank::world]

    def __len__(self):
        # Estimate from the current sizes (OOM reductions can make an epoch
        # longer; the LR schedule is driven by epoch progress, not this).
        total = 0
        world = self.num_replicas
        for k, v in self.bins.items():
            bs = self.batch_manager.get(k) * world
            if bs > 0:
                total += len(v) // bs + (1 if len(v) % bs >= world else 0)
        return max(1, total)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 root_path,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 OOD_data="Data/OOD_texts.txt",
                 min_length=50,
                 dynamic_batch=False,
                 multispeaker=True,
                 ):

        self.text_cleaner = TextCleaner()
        self.sr = sr
        self.multispeaker = multispeaker

        _data_list = [l.strip().split('|') for l in data_list if l.strip()]
        data_list = []
        n_long = 0
        for data in _data_list:
            data = list(data[:3]) if len(data) >= 3 else [*data, '0']
            # PL-BERT only has 512 positions; longer texts crash the model
            # ("index out of range in self"), so drop them up front.
            if len(self.text_cleaner(data[1])) > MAX_TEXT_TOKENS:
                n_long += 1
                continue
            data_list.append(tuple(data))
        if n_long:
            logger.warning('FilePathDataset: dropped %d samples with more than %d '
                           'phoneme tokens (PL-BERT limit); split them into '
                           'shorter utterances to use them', n_long, MAX_TEXT_TOKENS)
        self.data_list = data_list

        # speaker -> sample indices, for sampling style references quickly
        self.speaker_index = {}
        for i, data in enumerate(self.data_list):
            self.speaker_index.setdefault(str(data[2]), []).append(i)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192

        self.min_length = min_length
        with open(OOD_data, 'r', encoding='utf-8') as f:
            tl = f.readlines()
        idx = 1 if '.wav' in tl[0].split('|')[0] else 0
        ptexts = [t.split('|')[idx] for t in tl if len(t.split('|')) > idx]
        # OOD texts feed PL-BERT in SLM adversarial training: same 512 limit.
        self.ptexts = [t for t in ptexts
                       if len(self.text_cleaner(t)) <= MAX_TEXT_TOKENS]
        self._ood_long = [t for t in self.ptexts if len(t) >= self.min_length]

        self.root_path = root_path
        self.dynamic_batch = dynamic_batch
        self._warned_sr = False

        self.sample_bins = None
        if dynamic_batch:
            self._precompute_bins()

    def _precompute_bins(self):
        """Assign every sample to a length bin (reads file headers only;
        falls back to decoding when the header has no usable length)."""
        self.sample_bins = []
        n_header_fail = 0
        n_decode_fail = 0
        first_error = None
        for data in self.data_list:
            wave_path = data[0]
            full_path = osp.join(self.root_path, wave_path)
            n_samples = 0
            try:
                info = sf.info(full_path)
                n_samples = info.frames
                if n_samples and info.samplerate != self.sr:
                    n_samples = int(n_samples * self.sr / info.samplerate)
            except Exception as e:
                first_error = first_error or ('%s: %s' % (wave_path, e))

            if n_samples <= 0:
                n_header_fail += 1
                try:
                    wave, sr = sf.read(full_path)
                    n_samples = wave.shape[0]
                    if sr != self.sr:
                        n_samples = int(n_samples * self.sr / sr)
                except Exception as e:
                    n_decode_fail += 1
                    first_error = first_error or ('%s: %s' % (wave_path, e))
                    n_samples = 0
            self.sample_bins.append(get_time_bin(n_samples) if n_samples > 0 else None)

        if n_header_fail:
            logger.warning(
                'FilePathDataset: %d/%d files had no usable length in their '
                'header; fell back to decoding them (slower startup). First '
                'issue: %s', n_header_fail, len(self.data_list), first_error)
        if n_decode_fail:
            logger.error(
                'FilePathDataset: %d/%d files could not be read at all and '
                'will be skipped. Check root_path and file paths. First '
                'error: %s', n_decode_fail, len(self.data_list), first_error)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        path = data[0]

        if self.dynamic_batch:
            frame_count = get_frame_count(self.sample_bins[idx])
            wave, text_tensor, speaker_id = self._load_tensor(data, frame_count)
        else:
            wave, text_tensor, speaker_id = self._load_tensor(data)

        mel_tensor = preprocess(wave).squeeze()

        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]

        # Reference sample of the same speaker (only used by multi-speaker
        # models; single-speaker training skips the extra file read).
        if self.multispeaker:
            ref_idx = random.choice(self.speaker_index[str(data[2])])
            ref_mel_tensor, ref_label = self._load_data(self.data_list[ref_idx])
        else:
            ref_mel_tensor, ref_label = torch.zeros(80, 1), speaker_id

        # get OOD text
        pool = self._ood_long if self._ood_long else self.ptexts
        ps = pool[np.random.randint(0, len(pool))]
        text = self.text_cleaner(ps)
        text.insert(0, 0)
        text.append(0)
        ref_text = torch.LongTensor(text)

        return speaker_id, acoustic_feature, text_tensor, ref_text, ref_mel_tensor, ref_label, path, wave

    def _load_tensor(self, data, frame_count=None):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(osp.join(self.root_path, wave_path))
        if wave.ndim == 2:
            wave = wave[:, 0]
        if sr != self.sr:
            if not self._warned_sr:
                self._warned_sr = True
                logger.warning('Resampling %s from %d Hz to %d Hz on the fly (slow); '
                               'resample the dataset beforehand', wave_path, sr, self.sr)
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.sr)

        if frame_count is not None:
            # Pad symmetrically to exactly frame_count * hop_length samples, so
            # every item of a same-bin batch has the same length.
            target = frame_count * _HOP_LENGTH
            if len(wave) > target:
                wave = wave[:target]
            pad_total = target - len(wave)
            pad_start = pad_total // 2
            wave = np.concatenate([np.zeros(pad_start), wave,
                                   np.zeros(pad_total - pad_start)])
        else:
            wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)

        text = self.text_cleaner(text)

        text.insert(0, 0)
        text.append(0)

        text = torch.LongTensor(text)

        return wave, text, speaker_id

    def _load_data(self, data):
        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, speaker_id


class Collater(object):
    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave

    def __call__(self, batch):
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rtext_length = max([b[3].shape[0] for b in batch])

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        paths = ['' for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]

        for bid, (label, mel, text, ref_text, ref_mel, ref_label, path, wave) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel

            ref_labels[bid] = ref_label
            waves[bid] = wave

        return waves, texts, input_lengths, ref_texts, ref_lengths, mels, output_lengths, ref_mels


def build_dataloader(path_list,
                     root_path,
                     validation=False,
                     OOD_data="Data/OOD_texts.txt",
                     min_length=50,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={},
                     dynamic_batch=False,
                     batch_size_file='batch_sizes.json',
                     max_batch_frames=None,
                     max_batch_size=64,
                     num_replicas=1,
                     rank=0,
                     prefetch_factor=2,
                     persistent_workers=True):
    """Build a DataLoader for StyleTTS2 training or validation.

    With *dynamic_batch* (training only) samples are grouped into length bins
    and batched per bin by :class:`DynamicBatchSampler`; the batch size of
    each bin is managed by a :class:`BatchManager` (attached to the returned
    loader as ``.batch_manager``) whose frame budget is *max_batch_frames*
    (defaults to ``batch_size * 400``).  Otherwise this is the original
    random-batch loader (``.batch_manager`` is None).
    """
    use_dynamic = dynamic_batch and not validation
    dataset = FilePathDataset(
        path_list, root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        validation=validation,
        dynamic_batch=use_dynamic,
        **dataset_config,
    )
    collate_fn = Collater(**collate_config)

    # prefetch_factor/persistent_workers only apply when workers are spawned.
    extra_kwargs = {}
    if num_workers > 0:
        extra_kwargs['prefetch_factor'] = prefetch_factor
        extra_kwargs['persistent_workers'] = persistent_workers
    pin = (str(device) != 'cpu') and torch.cuda.is_available()

    if use_dynamic:
        batch_manager = BatchManager(batch_size_file,
                                     max_batch_frames=max_batch_frames or batch_size * 400,
                                     max_batch_size=max_batch_size,
                                     is_main_process=(rank == 0))
        sampler = DynamicBatchSampler(dataset, batch_manager, shuffle=True,
                                      num_replicas=num_replicas, rank=rank)
        data_loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin,
            **extra_kwargs,
        )
        data_loader.batch_manager = batch_manager
    else:
        sampler = None
        if num_replicas > 1 and not validation:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=num_replicas, rank=rank, shuffle=True, drop_last=True)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(not validation) and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=(not validation),
            collate_fn=collate_fn,
            pin_memory=pin,
            **extra_kwargs,
        )
        data_loader.batch_manager = None

    return data_loader
