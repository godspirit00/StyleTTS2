# StyleTTS2 → ONNX

Export a StyleTTS2 checkpoint to ONNX and run it with `onnxruntime`, including
the short-sentence workaround (append a filler sentence, then delete it via the
predicted durations).

Files:

| file | purpose |
|------|---------|
| `onnx_export.py` | checkpoint (`.pth` + `config.yml`) → 4 ONNX graphs + `meta.json` |
| `onnx_infer.py`  | `StyleTTS2ONNX` low-level runtime (numpy + onnxruntime) + CLI |
| `onnx_stft.py`   | ONNX-friendly STFT/iSTFT (replaces `torch.istft`, which is not exportable) |
| `test_parity.py` | validates the ONNX graphs against the original PyTorch model |

## Why 4 graphs instead of 1?

StyleTTS2 cannot be a single graph:

* the speaking style comes from an **iterative diffusion sampler**, and
* the short-sentence workaround needs to edit the **per-token durations**
  *between* duration prediction and audio synthesis.

So the pipeline is split exactly along those seams, mirroring `inference2()`:

```
1. encoder.onnx     tokens                      -> bert_dur, d_en, t_en
2. diffusion.onnx   x, sigma, bert_dur, scale   -> denoised     (called in a Python sampler loop)
3. predictor.onnx   d_en, s                     -> d, duration
   --- Python: round durations, apply workaround, build alignment matrix ---
4. decoder.onnx     d, t_en, s, ref, aln        -> waveform
```

The alignment matrix is built in Python (step between 3 and 4), which is exactly
where `styletts2_model_loader.py` deletes the appended filler by zeroing its
`pred_dur`. That is why **the workaround keeps working with ONNX** — see
`StyleTTS2ONNX.inference(..., additional_ph=...)` in `onnx_infer.py`.

## Export

```bash
python onnx_export.py \
  -c G:/tts/StyleTTS2-models/RasaEri/config_ft-single.yml \
  -w G:/tts/StyleTTS2-models/RasaEri/epoch_2nd_00018-Erinome_ft_model.pth \
  -o G:/tts/StyleTTS2-models/RasaEri/onnx
```

## Run

CLI (does its own phonemization + auto-filler for short inputs):

```bash
python onnx_infer.py -m .../onnx -t "Hello." -o out.wav
```

Library:

```python
from onnx_infer import StyleTTS2ONNX
m = StyleTTS2ONNX(".../onnx")
# phonemes = your IPA string (same tokenizer as the PyTorch model)
audio, s_prev = m.inference(phonemes,
                            additional_ph=filler_phonemes,  # None to disable
                            speed=1.0, diffusion_steps=5,
                            embedding_scale=1.0, seed=0)
```

`additional_ph` is the phonemized filler sentence appended to the end; its tokens
(and the joining space) are removed by zeroing their durations, identical to the
single-speaker `inference2()` path in the wrapper.

## Deterministic mode

The decoder's harmonic source uses a random initial phase and additive noise
(two `Random*Like` ops), so output varies run-to-run. Passing
`deterministic=True` (on `StyleTTS2ONNX` or `TTSModelONNX`, or `--deterministic`
on either CLI) rewrites those two nodes to deterministic zeros at load time, so
a given `(text, seed)` produces **bit-exact** output every run. Impact on
quality is negligible — the averaged magnitude spectrum matches the stochastic
output at correlation ≈ 0.999.

Note: onnxruntime advances its RNG across `run()` calls even with a fixed `seed`
attribute, which is why zeroing the nodes (rather than seeding them) is the
reliable route to reproducibility. For bit-exact results use the CPU provider;
GPU float reductions can differ in the last bits.

## Validation

`python test_parity.py` compares against the original PyTorch model:

* encoder / diffusion / predictor: max relative error ~1e-6 (deterministic).
* decoder: the harmonic source uses a **random initial phase**, so waveforms are
  not bit-identical across runs. The averaged magnitude spectrum matches the
  torch↔torch self-consistency baseline (spec corr ≈ 0.93 vs 0.95 baseline),
  confirming the deterministic path is reproduced.

## Notes / limitations

* Implemented for **single-speaker iSTFTNet** models (`multispeaker: false`),
  which is what the RasaEri checkpoints are. The multispeaker `inference()` path
  (reference-audio style via `StyleTransformer1d`) is not exported.
* `torch.istft` has no ONNX exporter, so `onnx_stft.CustomSTFT` reimplements
  STFT/iSTFT with conv/matmul (numerically matched to ~1e-6).
* `InstanceNorm(affine=False)` and `pack_padded_sequence` are not exportable;
  the export script swaps them for equivalent ops (valid for batch size 1).
* Sequence length is a dynamic axis, so one export handles any input length.
