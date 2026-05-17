#coding: utf-8
import os
import os.path as osp
import time
import random
import numpy as np
import random
import soundfile as sf
import librosa
import json

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import pandas as pd

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»"" '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
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
# Dynamic batching helpers
# ---------------------------------------------------------------------------
_HOP_LENGTH = 300  # must match SPECT_PARAMS hop_length

def get_time_bin(sample_count):
    """Map audio sample count to a bin index.

    Each bin covers 20 mel frames (~0.25 s at 24 kHz / hop 300).
    Bin 0 covers the shortest samples (fewer than 20 frames of audio).
    """
    frames = sample_count // _HOP_LENGTH
    if frames >= 20:
        return (frames - 20) // 20
    return 0

def get_frame_count(bin_id):
    """Return the padded mel-frame target for a given bin.

    Bin 0 → 60 frames, bin k → 60 + 20*k frames.
    The +40 headroom over the bin's lower boundary guarantees the actual
    audio always fits inside the target without truncation.
    """
    return bin_id * 20 + 60

def get_bin_from_mel_input_length(mel_input_length):
    """Recover the bin id from a batch's padded mel-frame tensor.

    With dynamic batching every sample in a batch has the same padded length
    equal to get_frame_count(bin_id) = 60 + 20*bin_id.  Inverting:
        bin_id = (frame_count - 60) // 20
    Safe to call even if mel_input_length has minor rounding; always >= 0.
    """
    frame_count = int(mel_input_length.min().item())
    return max(0, (frame_count - 60) // 20)


class BatchManager:
    """Persists per-bin batch sizes in a JSON file.

    Allows automatic reduction on OOM and manual tuning between runs.
    If the JSON does not exist, *default_batch_size* is used for every bin.
    """

    def __init__(self, json_path, default_batch_size=4):
        self.json_path = json_path
        self.default_batch_size = default_batch_size
        self.batch_sizes = {}
        if osp.exists(json_path):
            with open(json_path) as f:
                self.batch_sizes = {int(k): int(v) for k, v in json.load(f).items()}

    def get(self, bin_id):
        return max(1, self.batch_sizes.get(bin_id, self.default_batch_size))

    def set(self, bin_id, size):
        self.batch_sizes[bin_id] = max(1, size)
        self._save()

    def decrement(self, bin_id):
        """Reduce batch size for *bin_id* by 1.  Returns True if decremented."""
        cur = self.get(bin_id)
        if cur > 1:
            self.set(bin_id, cur - 1)
            logger.info('BatchManager: bin %d batch size reduced to %d', bin_id, cur - 1)
            return True
        return False

    def scale_all(self, factor):
        """Multiply every bin's batch size by *factor* (floor, minimum 1) and save.

        Call this proactively at epoch boundaries where VRAM usage jumps
        (e.g. when discriminators or the diffusion model join training) so that
        the next epoch starts with sizes that fit in the new memory budget.
        """
        all_bins = set(self.batch_sizes.keys())
        # Also cover bins that haven't been written yet (still at default)
        for bin_id in all_bins:
            new_size = max(1, int(self.get(bin_id) * factor))
            self.batch_sizes[bin_id] = new_size
        self._save()
        logger.info('BatchManager: all bin batch sizes scaled by %.2f', factor)

    def _save(self):
        with open(self.json_path, 'w') as f:
            json.dump({str(k): v for k, v in self.batch_sizes.items()}, f, indent=2)


class DynamicBatchSampler(torch.utils.data.Sampler):
    """Yields same-bin batches so every sample in a batch has the same length.

    This eliminates cross-sample clipping: because all items share the same
    padded length, the training loop's "clip to shortest" code becomes a no-op
    (random_start is always 0, mel_len equals the full sample length).
    """

    def __init__(self, dataset, batch_manager, shuffle=True):
        self.batch_manager = batch_manager
        self.shuffle = shuffle
        self.bins = {}
        for idx, bin_id in enumerate(dataset.sample_bins):
            self.bins.setdefault(bin_id, []).append(idx)

    def __iter__(self):
        bins = {k: list(v) for k, v in self.bins.items()}
        if self.shuffle:
            for indices in bins.values():
                random.shuffle(indices)

        batches = []
        for bin_id, indices in bins.items():
            bs = self.batch_manager.get(bin_id)
            # drop_last when shuffling to keep batch sizes uniform
            n = (len(indices) // bs) * bs if self.shuffle else len(indices)
            for i in range(0, n, bs):
                batches.append(indices[i:i + bs])

        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        return sum(
            len(v) // self.batch_manager.get(k)
            for k, v in self.bins.items()
        )


# ---------------------------------------------------------------------------
# Batch-size probe
# ---------------------------------------------------------------------------

def probe_batch_sizes(dataset, batch_manager, test_fn, vram_reserve_mb=512, logger=None):
    """Calibrate per-bin batch sizes before training starts.

    Iterates bins from shortest to longest.  For each bin calls *test_fn(batch)*
    (a full forward+backward training step) with the current batch size.  On OOM
    the batch size is decremented and the step is retried; bins that still OOM at
    batch_size=1 are set to 0 (skipped during training).  Results are persisted
    to ``batch_manager.json_path`` so subsequent runs reuse them.

    Args:
        dataset:          A :class:`FilePathDataset` with ``dynamic_batch=True``
                          (i.e. ``sample_bins`` already populated).
        batch_manager:    The :class:`BatchManager` whose sizes will be updated.
        test_fn:          ``callable(batch) -> None`` — runs one training step.
                          Should raise ``RuntimeError`` on OOM.
        vram_reserve_mb:  MiB to pre-allocate before probing so that estimates
                          are conservative (real training has other overhead).
        logger:           Optional logger for progress messages.
    """
    import gc
    from collections import defaultdict

    _log = (lambda msg: logger.info(msg)) if logger else (lambda msg: None)

    # Group indices by bin
    bins = defaultdict(list)
    for idx, bin_id in enumerate(dataset.sample_bins):
        bins[bin_id].append(idx)

    # Reserve VRAM so probe estimates are conservative
    reserve = None
    if vram_reserve_mb > 0:
        n_elems = vram_reserve_mb * 1024 * 1024 // 4  # float32
        try:
            reserve = torch.zeros(n_elems, dtype=torch.float32, device='cuda')
            _log(f'Probe: reserved {vram_reserve_mb} MiB of VRAM')
        except RuntimeError:
            _log('Probe: could not reserve VRAM, probing without reserve')

    collate_fn = Collater()

    for bin_id in sorted(bins.keys()):
        indices = bins[bin_id]
        while True:
            bs = batch_manager.get(bin_id)
            if bs == 0:
                _log(f'Probe: bin {bin_id} marked as skip')
                break
            if len(indices) < bs:
                _log(f'Probe: bin {bin_id} has fewer samples than batch_size={bs}, keeping as-is')
                break
            try:
                batch = collate_fn([dataset[i] for i in indices[:bs]])
                test_fn(batch)
                _log(f'Probe: bin {bin_id} -> batch_size={bs} OK')
                break
            except RuntimeError as e:
                if 'out of memory' not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                gc.collect()
                _log(f'Probe: bin {bin_id} OOM at batch_size={bs}')
                if not batch_manager.decrement(bin_id):
                    # Already at 1 and still OOM — skip this bin
                    batch_manager.set(bin_id, 0)
                    _log(f'Probe: bin {bin_id} skipped (batch_size=1 still OOMs)')
                    break

    if reserve is not None:
        del reserve
        torch.cuda.empty_cache()
        gc.collect()

    batch_manager._save()
    _log('Probe complete — batch_sizes.json updated')


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
                 ):

        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS

        _data_list = [l.strip().split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner()
        self.sr = sr

        self.df = pd.DataFrame(self.data_list)

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192

        self.min_length = min_length
        with open(OOD_data, 'r', encoding='utf-8') as f:
            tl = f.readlines()
        idx = 1 if '.wav' in tl[0].split('|')[0] else 0
        self.ptexts = [t.split('|')[idx] for t in tl]

        self.root_path = root_path
        self.dynamic_batch = dynamic_batch

        if dynamic_batch:
            self._precompute_bins()

    def _precompute_bins(self):
        """Read audio file headers (no decode) and assign each sample to a bin."""
        self.sample_bins = []
        for data in self.data_list:
            wave_path = data[0]
            try:
                info = sf.info(osp.join(self.root_path, wave_path))
                n_samples = info.frames
                if info.samplerate != self.sr:
                    n_samples = int(n_samples * self.sr / info.samplerate)
            except Exception:
                n_samples = 0
            self.sample_bins.append(get_time_bin(n_samples))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        path = data[0]

        if self.dynamic_batch:
            bin_id = self.sample_bins[idx]
            frame_count = get_frame_count(bin_id)
            wave, text_tensor, speaker_id = self._load_tensor(data, frame_count)
        else:
            wave, text_tensor, speaker_id = self._load_tensor(data)

        mel_tensor = preprocess(wave).squeeze()

        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]

        # get reference sample
        ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
        ref_mel_tensor, ref_label = self._load_data(ref_data[:3])

        # get OOD text
        ps = ""

        while len(ps) < self.min_length:
            rand_idx = np.random.randint(0, len(self.ptexts) - 1)
            ps = self.ptexts[rand_idx]

            text = self.text_cleaner(ps)
            text.insert(0, 0)
            text.append(0)

            ref_text = torch.LongTensor(text)

        return speaker_id, acoustic_feature, text_tensor, ref_text, ref_mel_tensor, ref_label, path, wave

    def _load_tensor(self, data, frame_count=None):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(osp.join(self.root_path, wave_path))
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
            print(wave_path, sr)

        if frame_count is not None:
            # Pad symmetrically to exactly frame_count * hop_length samples.
            # Because all samples in the same bin are padded to the same length,
            # the collater and training loop never need to clip across samples.
            target = frame_count * _HOP_LENGTH
            if len(wave) > target:
                wave = wave[:target]
            else:
                pad_total = target - len(wave)
                pad_start = pad_total // 2
                wave = np.concatenate([
                    np.zeros(pad_start),
                    wave,
                    np.zeros(pad_total - pad_start),
                ])
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
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave


    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
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
                     batch_size_file='batch_sizes.json'):
    """Build a DataLoader for StyleTTS2 training or validation.

    When *dynamic_batch* is True the dataset pre-scans audio lengths, groups
    samples into fixed-length bins, and uses :class:`DynamicBatchSampler` so
    that every item in a batch has the same padded length.  This removes the
    need to clip all samples to the shortest one in the batch (the clipping
    problem present in the original code).

    Per-bin batch sizes are read from *batch_size_file* (JSON).  If the file
    does not exist, *batch_size* is used as the default for every bin.
    To handle OOM, reduce a bin's entry in the JSON and restart, or let the
    training script call ``train_dataloader.batch_manager.decrement(bin_id)``.
    """
    dataset = FilePathDataset(
        path_list, root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        validation=validation,
        dynamic_batch=dynamic_batch and (not validation),
        **dataset_config,
    )
    collate_fn = Collater(**collate_config)

    if dynamic_batch and not validation:
        batch_manager = BatchManager(batch_size_file, default_batch_size=batch_size)
        sampler = DynamicBatchSampler(dataset, batch_manager, shuffle=True)
        data_loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=(device != 'cpu'),
        )
        # Attach batch_manager so training scripts can access it for OOM handling
        data_loader.batch_manager = batch_manager
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(not validation),
            num_workers=num_workers,
            drop_last=(not validation),
            collate_fn=collate_fn,
            pin_memory=(device != 'cpu'),
        )
        data_loader.batch_manager = None

    return data_loader
