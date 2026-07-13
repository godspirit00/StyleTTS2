"""
Validate the exported ONNX graphs against the ORIGINAL (unpatched) PyTorch
model.  Encoder / diffusion / predictor are deterministic and compared exactly.
The decoder contains random noise excitation, so we compare the ONNX-vs-torch
difference against the torch-vs-torch self difference (two independent noise
draws): if they are the same order of magnitude, the deterministic path matches
and only the noise realisation differs.
"""
import os
import numpy as np
import torch
import onnxruntime as ort

import onnx_export as ox

MODEL_DIR = "G:/tts/StyleTTS2-models/RasaEri/onnx"
CONFIG = "G:/tts/StyleTTS2-models/RasaEri/config_ft-single.yml"
WEIGHTS = "G:/tts/StyleTTS2-models/RasaEri/epoch_2nd_00018-Erinome_ft_model.pth"

torch.manual_seed(1)
np.random.seed(1)

# original model (no export monkeypatches applied)
model, config, plbert_config = ox.build_model(CONFIG, WEIGHTS, "Utils/PLBERT")
style_dim = config.model_params.style_dim

prov = ["CPUExecutionProvider"]
enc_s = ort.InferenceSession(os.path.join(MODEL_DIR, "encoder.onnx"), providers=prov)
dif_s = ort.InferenceSession(os.path.join(MODEL_DIR, "diffusion.onnx"), providers=prov)
prd_s = ort.InferenceSession(os.path.join(MODEL_DIR, "predictor.onnx"), providers=prov)
dec_s = ort.InferenceSession(os.path.join(MODEL_DIR, "decoder.onnx"), providers=prov)


def rel(a, b):
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


tokens = torch.randint(1, 170, (1, 25), dtype=torch.long)
il = torch.LongTensor([tokens.shape[-1]])
tm = length_to_mask(il)

with torch.no_grad():
    t_en = model.text_encoder(tokens, il, tm)
    bert_dur = model.bert(tokens, attention_mask=(~tm).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

o_bert, o_den, o_ten = enc_s.run(None, {"tokens": tokens.numpy()})
print("[encoder] bert_dur rel err:", rel(o_bert, bert_dur.numpy()))
print("[encoder] d_en     rel err:", rel(o_den, d_en.numpy()))
print("[encoder] t_en     rel err:", rel(o_ten, t_en.numpy()))

# ---- diffusion denoise step (deterministic given inputs) ----
x = torch.randn(1, 1, style_dim * 2)
sigma = torch.tensor([0.85])
scale = 1.0
with torch.no_grad():
    kd = model.diffusion.diffusion
    net = kd.net
    c_skip, c_out, c_in, c_noise = kd.get_scale_weights(sigma)
    fixed = net.fixed_embedding(bert_dur)
    out = net.run(c_in * x, c_noise, embedding=bert_dur, features=None)
    out_m = net.run(c_in * x, c_noise, embedding=fixed, features=None)
    ref_denoised = c_skip * x + c_out * (out_m + (out - out_m) * scale)
o_den2 = dif_s.run(None, {"x": x.numpy(), "sigma": sigma.numpy().astype(np.float32),
                          "bert_dur": bert_dur.numpy(),
                          "embedding_scale": np.array([scale], np.float32)})[0]
print("[diffusion] denoised rel err:", rel(o_den2, ref_denoised.numpy()))

# ---- predictor (deterministic) ----
s = torch.randn(1, style_dim)
with torch.no_grad():
    d = model.predictor.text_encoder(d_en, s, il, tm)
    xx, _ = model.predictor.lstm(d)
    dur = model.predictor.duration_proj(xx)
    dur = torch.sigmoid(dur).sum(dim=-1)
o_d, o_dur = prd_s.run(None, {"d_en": d_en.numpy(), "s": s.numpy()})
print("[predictor] d        rel err:", rel(o_d, d.numpy()))
print("[predictor] duration rel err:", rel(o_dur, dur.numpy()))

# ---- decoder (noise self-diff vs cross-diff) ----
ref = torch.randn(1, style_dim)
pred_dur = torch.clamp(torch.round(dur.squeeze()), min=1).long()
T = int(pred_dur.sum())
n = tokens.shape[-1]
aln = torch.zeros(1, n, T)
c = 0
for i in range(n):
    aln[0, i, c:c + int(pred_dur[i])] = 1
    c += int(pred_dur[i])

with torch.no_grad():
    en = d.transpose(-1, -2) @ aln
    F0, N = model.predictor.F0Ntrain(en, s)
    asr = t_en @ aln
    torch.manual_seed(10)
    w1 = model.decoder(asr, F0, N, ref).squeeze().numpy()
    torch.manual_seed(20)
    w2 = model.decoder(asr, F0, N, ref).squeeze().numpy()

o_audio = dec_s.run(None, {"d": d.numpy(), "t_en": t_en.numpy(), "s": s.numpy(),
                           "ref": ref.numpy(), "aln": aln.numpy()})[0][0]

L = min(len(w1), len(w2), len(o_audio))
w1, w2, o_audio = w1[:L], w2[:L], o_audio[:L]
self_diff = np.abs(w1 - w2).mean()
cross_diff = np.abs(o_audio - w1).mean()
corr = np.corrcoef(o_audio, w1)[0, 1]
print("[decoder] torch RMS:", float(np.sqrt((w1 ** 2).mean())),
      " onnx RMS:", float(np.sqrt((o_audio ** 2).mean())))
print("[decoder] torch-self mean|diff|:", self_diff,
      " onnx-torch mean|diff|:", cross_diff)
print("[decoder] onnx-vs-torch corr:", corr)
