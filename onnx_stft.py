"""
ONNX-friendly STFT / iSTFT that numerically matches torch.stft / torch.istft
(center=True, onesided=True, normalized=False), implemented with plain
matmul + conv_transpose1d so it can be traced and exported to ONNX.

Only the operations used by StyleTTS2's iSTFTNet decoder are implemented:
  - transform(x)          -> (magnitude, phase)     [like TorchSTFT.transform]
  - inverse(mag, phase)   -> waveform (B, 1, T)     [like TorchSTFT.inverse]

The default torch behaviour we replicate:
  torch.stft(x, n_fft, hop, win_length, window, center=True, return_complex=True)
  torch.istft(spec, n_fft, hop, win_length, window, center=True)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window


class CustomSTFT(nn.Module):
    def __init__(self, filter_length=20, hop_length=5, win_length=20, window="hann"):
        super().__init__()
        self.filter_length = int(filter_length)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.pad = self.filter_length // 2          # center padding
        n_freq = self.filter_length // 2 + 1

        win = get_window(window, self.win_length, fftbins=True).astype(np.float64)
        # If win_length < filter_length torch pads the window with zeros (centred).
        if self.win_length < self.filter_length:
            left = (self.filter_length - self.win_length) // 2
            padded = np.zeros(self.filter_length, dtype=np.float64)
            padded[left:left + self.win_length] = win
            win = padded
        window_t = torch.from_numpy(win).float()

        n = np.arange(self.filter_length)
        k = np.arange(n_freq)
        angle = 2.0 * np.pi * np.outer(k, n) / self.filter_length      # (n_freq, n_fft)

        # ---- forward DFT basis (analysis), window applied here ----
        cos_b = np.cos(angle) * win[None, :]
        sin_b = np.sin(angle) * win[None, :]
        # weight shape for conv1d: (out_channels=n_freq, in_channels=1, kernel=n_fft)
        self.register_buffer("fwd_cos", torch.from_numpy(cos_b).float().unsqueeze(1))
        self.register_buffer("fwd_sin", torch.from_numpy(sin_b).float().unsqueeze(1))

        # ---- inverse DFT basis (synthesis) ----
        # frame[n] = (1/N) sum_k w_k (Re[k] cos - Im[k] sin), w_k = 2 except k=0,Nyq
        w = np.full(n_freq, 2.0)
        w[0] = 1.0
        if self.filter_length % 2 == 0:
            w[-1] = 1.0
        idft_cos = (w[:, None] / self.filter_length) * np.cos(angle)   # (n_freq, n_fft)
        idft_sin = -(w[:, None] / self.filter_length) * np.sin(angle)
        # apply synthesis window
        idft_cos = idft_cos * win[None, :]
        idft_sin = idft_sin * win[None, :]
        self.register_buffer("inv_cos", torch.from_numpy(idft_cos).float())
        self.register_buffer("inv_sin", torch.from_numpy(idft_sin).float())

        # overlap-add identity kernel: (in=n_fft, out=1, kernel=n_fft)
        eye = torch.eye(self.filter_length).unsqueeze(1)               # (n_fft,1,n_fft)
        self.register_buffer("ola_kernel", eye)
        self.register_buffer("window_sq", window_t ** 2)

    def transform(self, x):
        # x: (B, T)
        x = x.unsqueeze(1)                                            # (B,1,T)
        x = F.pad(x, (self.pad, self.pad), mode="reflect")
        real = F.conv1d(x, self.fwd_cos, stride=self.hop_length)      # (B,n_freq,F)
        imag = -F.conv1d(x, self.fwd_sin, stride=self.hop_length)
        mag = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-12))
        phase = torch.atan2(imag, real)
        return mag, phase

    def inverse(self, magnitude, phase):
        # magnitude/phase: (B, n_freq, F)
        real = magnitude * torch.cos(phase)                          # (B,n_freq,F)
        imag = magnitude * torch.sin(phase)
        # frames: (B, n_fft, F)  -> use as conv_transpose input (channels=n_fft)
        frames = (torch.einsum("bkf,kn->bnf", real, self.inv_cos)
                  + torch.einsum("bkf,kn->bnf", imag, self.inv_sin))
        signal = F.conv_transpose1d(frames, self.ola_kernel, stride=self.hop_length)  # (B,1,L)

        # window overlap-add normalisation, matching torch.istft
        F_frames = frames.shape[-1]
        win_sq = self.window_sq.view(1, self.filter_length, 1).expand(1, self.filter_length, F_frames)
        norm = F.conv_transpose1d(win_sq, self.ola_kernel, stride=self.hop_length)    # (1,1,L)
        signal = signal / (norm + 1e-11)

        # trim center padding
        signal = signal[..., self.pad:signal.shape[-1] - self.pad]
        return signal                                                # (B,1,T)

    def forward(self, x):
        mag, phase = self.transform(x)
        return self.inverse(mag, phase)


if __name__ == "__main__":
    torch.manual_seed(0)
    n_fft, hop, win = 20, 5, 20
    window = torch.from_numpy(get_window("hann", win, fftbins=True).astype(np.float32))

    x = torch.randn(2, 733)

    # reference forward
    ref = torch.stft(x, n_fft, hop, win, window=window, return_complex=True)
    ref_mag, ref_phase = torch.abs(ref), torch.angle(ref)

    cs = CustomSTFT(n_fft, hop, win)
    mag, phase = cs.transform(x)
    print("mag  max abs err:", (mag - ref_mag).abs().max().item())
    # compare complex (phase wraps, so compare real/imag)
    r_ref, i_ref = ref.real, ref.imag
    r, i = mag * torch.cos(phase), mag * torch.sin(phase)
    print("real max abs err:", (r - r_ref).abs().max().item())
    print("imag max abs err:", (i - i_ref).abs().max().item())

    # reference inverse (feed a network-like spectrum)
    spec = torch.exp(torch.randn(2, n_fft // 2 + 1, 40) * 0.3)
    ph = torch.sin(torch.randn(2, n_fft // 2 + 1, 40))
    ref_wave = torch.istft(spec * torch.exp(ph * 1j), n_fft, hop, win, window=window)
    my_wave = cs.inverse(spec, ph).squeeze(1)
    print("istft shapes:", ref_wave.shape, my_wave.shape)
    print("istft max abs err:", (ref_wave - my_wave).abs().max().item())
    print("istft rel err:", ((ref_wave - my_wave).abs().max() / ref_wave.abs().max()).item())
