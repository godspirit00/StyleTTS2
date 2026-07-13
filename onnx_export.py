"""
Export a StyleTTS2 checkpoint to a set of ONNX graphs.

StyleTTS2 cannot be exported as a single ONNX graph, mainly because:
  * the style is produced by an iterative diffusion sampler, and
  * the short-sentence workaround needs to manipulate the per-token
    durations *between* duration prediction and audio synthesis.

We therefore split the model into 4 ONNX graphs that mirror the reference
`inference2()` (single-speaker) pipeline:

  1. encoder.onnx     tokens                      -> bert_dur, d_en, t_en
  2. diffusion.onnx   x, sigma, bert_dur, scale   -> denoised          (called in a loop)
  3. predictor.onnx   d_en, s                     -> d, duration
  4. decoder.onnx     d, t_en, s, ref, aln        -> waveform

The Python runtime (onnx_infer.py) drives the diffusion sampler loop and,
crucially, builds the alignment matrix from the predicted durations.  That is
exactly where the "append a sentence then delete it via pred_dur" workaround
lives, so it keeps working with the ONNX models.

Usage:
  python onnx_export.py -c config.yml -w checkpoint.pth -o onnx_out
"""

import os
import sys
import json
import types
import argparse
import importlib
from collections import OrderedDict

import yaml
import torch
import torch.nn as nn
from munch import Munch
from transformers import AlbertConfig


# models.py uses package-relative imports ("from .Utils.ASR ..."), so the repo
# must be importable as a package. Register it under the name "StyleTTS2" (the
# same convention the project's own wrapper uses).
def _register_pkg():
    repo = os.path.dirname(os.path.abspath(__file__))
    if "StyleTTS2" not in sys.modules:
        pkg = types.ModuleType("StyleTTS2")
        pkg.__path__ = [repo]
        sys.modules["StyleTTS2"] = pkg
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return repo


_register_pkg()

models = importlib.import_module("StyleTTS2.models")
TextEncoder = models.TextEncoder
ProsodyPredictor = models.ProsodyPredictor
StyleEncoder = models.StyleEncoder
StyleTransformer1d = models.StyleTransformer1d
Transformer1d = models.Transformer1d
AudioDiffusionConditional = models.AudioDiffusionConditional
KDiffusion = models.KDiffusion
LogNormalDistribution = models.LogNormalDistribution
PlBert = importlib.import_module("StyleTTS2.Utils.PLBERT.util").CustomAlbert
from onnx_stft import CustomSTFT


# --------------------------------------------------------------------------- #
#  pack_padded_sequence is not ONNX-exportable. For batch size 1 with a fully
#  valid (unpadded) sequence the pack/pad round-trip is an identity, so we
#  temporarily replace it with a pass-through during export.
# --------------------------------------------------------------------------- #
def _patch_rnn_pack():
    orig_pack = nn.utils.rnn.pack_padded_sequence
    orig_pad = nn.utils.rnn.pad_packed_sequence

    def fake_pack(input, lengths, batch_first=False, enforce_sorted=True):
        return input

    def fake_pad(sequence, batch_first=False, padding_value=0.0,
                 total_length=None):
        return sequence, None

    nn.utils.rnn.pack_padded_sequence = fake_pack
    nn.utils.rnn.pad_packed_sequence = fake_pad
    return orig_pack, orig_pad


def _patch_duration_encoder():
    """DurationEncoder uses `s.permute(1, -1, 0)`; ONNX Transpose rejects the
    negative axis. Replace the method with an identical one using axis 2."""
    import torch.nn.functional as F
    DurationEncoder = models.DurationEncoder
    AdaLayerNorm = models.AdaLayerNorm

    def forward(self, x, style, text_lengths, m):
        masks = m.to(text_lengths.device)
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        input_lengths = text_lengths.cpu().numpy()
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)   # was permute(1, -1, 0)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])
                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad.to(x.device)
        return x.transpose(-1, -2)

    DurationEncoder.forward = forward


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    return d


def build_model(config_path, weights_path, plbert_dir):
    config = recursive_munch(yaml.safe_load(open(config_path, encoding="utf-8")))

    if "plbert_params" in config:
        plbert_config = config.plbert_params
    else:
        bert_path = config.get("PLBERT_dir", False)
        if not bert_path or not os.path.exists(bert_path):
            bert_path = plbert_dir
        plbert_config = recursive_munch(
            yaml.safe_load(open(os.path.join(bert_path, "config.yml"), encoding="utf-8"))
        ).model_params

    args = config.model_params
    plbert = PlBert(AlbertConfig(**plbert_config))
    plbert_encoder = nn.Linear(plbert_config.hidden_size, args.hidden_dim)

    if args.decoder.type == "istftnet":
        from StyleTTS2.Modules.istftnet import Decoder
        decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                          resblock_kernel_sizes=args.decoder.resblock_kernel_sizes,
                          upsample_rates=args.decoder.upsample_rates,
                          upsample_initial_channel=args.decoder.upsample_initial_channel,
                          resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                          upsample_kernel_sizes=args.decoder.upsample_kernel_sizes,
                          gen_istft_n_fft=args.decoder.gen_istft_n_fft,
                          gen_istft_hop_size=args.decoder.gen_istft_hop_size)
    else:
        from StyleTTS2.Modules.hifigan import Decoder
        decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                          resblock_kernel_sizes=args.decoder.resblock_kernel_sizes,
                          upsample_rates=args.decoder.upsample_rates,
                          upsample_initial_channel=args.decoder.upsample_initial_channel,
                          resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                          upsample_kernel_sizes=args.decoder.upsample_kernel_sizes)

    text_encoder = TextEncoder(channels=args.hidden_dim, kernel_size=5,
                               depth=args.n_layer, n_symbols=args.n_token)
    predictor = ProsodyPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim,
                                 nlayers=args.n_layer, max_dur=args.max_dur, dropout=args.dropout)
    style_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim)
    predictor_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim)

    if args.multispeaker:
        transformer = StyleTransformer1d(channels=args.style_dim * 2,
                                         context_embedding_features=plbert_config.hidden_size,
                                         context_features=args.style_dim * 2,
                                         **args.diffusion.transformer)
    else:
        transformer = Transformer1d(channels=args.style_dim * 2,
                                    context_embedding_features=plbert_config.hidden_size,
                                    **args.diffusion.transformer)

    diffusion = AudioDiffusionConditional(
        in_channels=1,
        embedding_max_length=plbert_config.max_position_embeddings,
        embedding_features=plbert_config.hidden_size,
        embedding_mask_proba=args.diffusion.embedding_mask_proba,
        channels=args.style_dim * 2,
        context_features=args.style_dim * 2,
    )
    diffusion.diffusion = KDiffusion(
        net=diffusion.unet,
        sigma_distribution=LogNormalDistribution(mean=args.diffusion.dist.mean, std=args.diffusion.dist.std),
        sigma_data=args.diffusion.dist.sigma_data,
        dynamic_threshold=0.0,
    )
    diffusion.diffusion.net = transformer
    diffusion.unet = transformer

    model = Munch(bert=plbert, bert_encoder=plbert_encoder, predictor=predictor,
                  decoder=decoder, text_encoder=text_encoder,
                  predictor_encoder=predictor_encoder, style_encoder=style_encoder,
                  diffusion=diffusion)

    params_whole = torch.load(weights_path, map_location="cpu", weights_only=True)
    params = params_whole["net"] if "net" in params_whole else params_whole
    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key])
            except Exception:
                sd = OrderedDict((k[7:], v) for k, v in params[key].items())
                model[key].load_state_dict(sd, strict=False)
            print("%s loaded" % key)

    for key in model:
        model[key].eval()
        model[key].to("cpu")

    return model, config, plbert_config


# --------------------------------------------------------------------------- #
#  InstanceNorm(affine=False) fails to export when ONNX shape inference loses
#  the channel dimension (which happens after the generator's reflection pad).
#  Replace every InstanceNorm with a manual, shape-agnostic equivalent.
# --------------------------------------------------------------------------- #
class ManualInstanceNorm(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.eps = orig.eps
        self.affine = orig.affine
        if orig.affine:
            self.weight = orig.weight
            self.bias = orig.bias

    def forward(self, x):
        dims = tuple(range(2, x.dim()))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        y = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = y * self.weight.view(shape) + self.bias.view(shape)
        return y


def replace_instancenorm(module):
    for name, child in module.named_children():
        if isinstance(child, (nn.InstanceNorm1d, nn.InstanceNorm2d)):
            setattr(module, name, ManualInstanceNorm(child))
        else:
            replace_instancenorm(child)


# --------------------------------------------------------------------------- #
#  ONNX sub-graph wrappers
# --------------------------------------------------------------------------- #
class EncoderONNX(nn.Module):
    """tokens -> bert_dur, d_en, t_en"""
    def __init__(self, model):
        super().__init__()
        self.text_encoder = model.text_encoder
        self.bert = model.bert
        self.bert_encoder = model.bert_encoder

    def forward(self, tokens):
        n = tokens.shape[-1]
        text_mask = torch.zeros(1, n, dtype=torch.bool, device=tokens.device)
        input_lengths = (~text_mask).sum(-1)
        t_en = self.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = self.bert(tokens, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        return bert_dur, d_en, t_en


class DiffusionONNX(nn.Module):
    """One denoise evaluation with classifier-free guidance (scale is an input)."""
    def __init__(self, model):
        super().__init__()
        self.kdiff = model.diffusion.diffusion
        self.net = model.diffusion.diffusion.net

    def forward(self, x_noisy, sigma, embedding, embedding_scale):
        c_skip, c_out, c_in, c_noise = self.kdiff.get_scale_weights(sigma)
        x_in = c_in * x_noisy
        fixed = self.net.fixed_embedding(embedding)
        out = self.net.run(x_in, c_noise, embedding=embedding, features=None)
        out_masked = self.net.run(x_in, c_noise, embedding=fixed, features=None)
        x_pred = out_masked + (out - out_masked) * embedding_scale
        return c_skip * x_noisy + c_out * x_pred


class PredictorONNX(nn.Module):
    """d_en, s -> d, duration (summed sigmoid, pre-rounding)."""
    def __init__(self, model):
        super().__init__()
        self.predictor = model.predictor

    def forward(self, d_en, s):
        n = d_en.shape[-1]
        text_mask = torch.zeros(1, n, dtype=torch.bool, device=d_en.device)
        input_lengths = (~text_mask).sum(-1)
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(dim=-1)
        return d, duration


class DecoderONNX(nn.Module):
    """d, t_en, s, ref, aln -> waveform."""
    def __init__(self, model, hifigan_shift):
        super().__init__()
        self.predictor = model.predictor
        self.decoder = model.decoder
        self.hifigan_shift = hifigan_shift

    @staticmethod
    def _shift(x):
        y = torch.zeros_like(x)
        y[:, :, 0] = x[:, :, 0]
        y[:, :, 1:] = x[:, :, 0:-1]
        return y

    def forward(self, d, t_en, s, ref, aln):
        en = d.transpose(-1, -2) @ aln
        if self.hifigan_shift:
            en = self._shift(en)
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        asr = t_en @ aln
        if self.hifigan_shift:
            asr = self._shift(asr)
        out = self.decoder(asr, F0_pred, N_pred, ref)
        return out.squeeze(1)


def export(config_path, weights_path, out_dir, plbert_dir, opset=17):
    os.makedirs(out_dir, exist_ok=True)
    _patch_rnn_pack()
    _patch_duration_encoder()
    model, config, plbert_config = build_model(config_path, weights_path, plbert_dir)
    for key in model:
        replace_instancenorm(model[key])

    args = config.model_params
    decoder_type = args.decoder.type
    hifigan_shift = (decoder_type == "hifigan")

    # replace the iSTFT with an ONNX-friendly equivalent
    gen = model.decoder.generator
    model.decoder.generator.stft = CustomSTFT(
        filter_length=gen.stft.filter_length,
        hop_length=gen.stft.hop_length,
        win_length=gen.stft.win_length,
    )

    style_dim = args.style_dim               # 128
    hidden = args.hidden_dim                 # 512
    bert_hidden = plbert_config.hidden_size  # 768
    dur_hidden = hidden + style_dim          # DurationEncoder output dim (640)
    n = 20                                    # dummy sequence length for tracing

    torch.manual_seed(0)
    tokens = torch.randint(1, args.n_token, (1, n), dtype=torch.long)

    # ---- 1. encoder ----
    enc = EncoderONNX(model).eval()
    with torch.no_grad():
        bert_dur, d_en, t_en = enc(tokens)
    torch.onnx.export(
        enc, (tokens,), os.path.join(out_dir, "encoder.onnx"),
        input_names=["tokens"], output_names=["bert_dur", "d_en", "t_en"],
        dynamic_axes={"tokens": {1: "N"}, "bert_dur": {1: "N"},
                      "d_en": {2: "N"}, "t_en": {2: "N"}},
        opset_version=opset, do_constant_folding=True)
    print("exported encoder.onnx")

    # ---- 2. diffusion ----
    diff = DiffusionONNX(model).eval()
    x_noisy = torch.randn(1, 1, style_dim * 2)
    sigma = torch.tensor([1.0], dtype=torch.float32)
    scale = torch.tensor([1.0], dtype=torch.float32)
    torch.onnx.export(
        diff, (x_noisy, sigma, bert_dur, scale), os.path.join(out_dir, "diffusion.onnx"),
        input_names=["x", "sigma", "bert_dur", "embedding_scale"],
        output_names=["denoised"],
        dynamic_axes={"bert_dur": {1: "N"}},
        opset_version=opset, do_constant_folding=True)
    print("exported diffusion.onnx")

    # ---- 3. predictor ----
    s = torch.randn(1, style_dim)
    pred = PredictorONNX(model).eval()
    with torch.no_grad():
        d, duration = pred(d_en, s)
    torch.onnx.export(
        pred, (d_en, s), os.path.join(out_dir, "predictor.onnx"),
        input_names=["d_en", "s"], output_names=["d", "duration"],
        dynamic_axes={"d_en": {2: "N"}, "d": {1: "N"}, "duration": {1: "N"}},
        opset_version=opset, do_constant_folding=True)
    print("exported predictor.onnx")

    # ---- 4. decoder ----
    T = 60
    ref = torch.randn(1, style_dim)
    aln = torch.zeros(1, n, T)
    for i in range(n):
        aln[0, i, (i * T // n):((i + 1) * T // n)] = 1
    dec = DecoderONNX(model, hifigan_shift).eval()
    torch.onnx.export(
        dec, (d, t_en, s, ref, aln), os.path.join(out_dir, "decoder.onnx"),
        input_names=["d", "t_en", "s", "ref", "aln"], output_names=["audio"],
        dynamic_axes={"d": {1: "N"}, "t_en": {2: "N"}, "aln": {1: "N", 2: "T"},
                      "audio": {1: "L"}},
        opset_version=opset, do_constant_folding=True)
    print("exported decoder.onnx")

    meta = {
        "multispeaker": bool(args.multispeaker),
        "decoder_type": decoder_type,
        "sample_rate": 24000,
        "style_dim": int(style_dim),
        "hidden_dim": int(hidden),
        "bert_hidden": int(bert_hidden),
        "dur_hidden": int(dur_hidden),
        "n_token": int(args.n_token),
        "noise_dim": int(style_dim * 2),
        "max_dur": int(args.max_dur),
        "diffusion": {
            "sigma_min": 0.0001,
            "sigma_max": 3.0,
            "rho": 9.0,
            "default_steps": 5,
            "default_embedding_scale": 1.0,
        },
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("wrote meta.json")
    print("Done. ONNX graphs written to", out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser("Export StyleTTS2 checkpoint to ONNX")
    p.add_argument("-c", required=True, help="path to config yml")
    p.add_argument("-w", required=True, help="path to checkpoint .pth")
    p.add_argument("-o", default="onnx_out", help="output directory")
    p.add_argument("--plbert", default="Utils/PLBERT", help="fallback PLBERT dir")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()
    export(args.c, args.w, args.o, args.plbert, args.opset)
