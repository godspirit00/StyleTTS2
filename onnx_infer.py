"""
Run a StyleTTS2 model that was exported with onnx_export.py.

This reimplements the single-speaker `inference2()` pipeline on top of
onnxruntime, and keeps the "short-sentence workaround": a filler sentence is
appended to the phonemes and then removed again by zeroing its predicted
durations before the alignment matrix is built.  Because the alignment is built
in Python (between the `predictor` and `decoder` ONNX graphs), the workaround
works exactly like it does with the PyTorch wrapper (styletts2_model_loader.py).

Only numpy + onnxruntime are needed at run time (plus phonemizer/espeak and the
project's TextCleaner for turning text into tokens).

Example:
  python onnx_infer.py -m onnx_out -t "Hello." -o out.wav
"""

import os
import sys
import json
import types
import argparse

import numpy as np
import onnx
import onnxruntime as ort


# --------------------------------------------------------------------------- #
#  Make the project importable as the "StyleTTS2" package (for TextCleaner).
# --------------------------------------------------------------------------- #
def _register_pkg():
    repo = os.path.dirname(os.path.abspath(__file__))
    if "StyleTTS2" not in sys.modules:
        pkg = types.ModuleType("StyleTTS2")
        pkg.__path__ = [repo]
        sys.modules["StyleTTS2"] = pkg
    if repo not in sys.path:
        sys.path.insert(0, repo)


_register_pkg()
from StyleTTS2.text_utils import TextCleaner


# --------------------------------------------------------------------------- #
#  Diffusion sampler (ADPM2 + Karras schedule), ported to numpy.
#  Mirrors Modules/diffusion/sampler.py exactly for single-speaker models.
# --------------------------------------------------------------------------- #
def karras_schedule(num_steps, sigma_min, sigma_max, rho):
    rho_inv = 1.0 / rho
    steps = np.arange(num_steps, dtype=np.float32)
    sigmas = (sigma_max ** rho_inv + (steps / (num_steps - 1)) *
              (sigma_min ** rho_inv - sigma_max ** rho_inv)) ** rho
    sigmas = np.concatenate([sigmas, np.zeros(1, dtype=np.float32)])
    return sigmas


def _adpm2_sigmas(sigma, sigma_next):
    sigma_up = np.sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
    sigma_down = np.sqrt(sigma_next ** 2 - sigma_up ** 2)
    sigma_mid = ((sigma ** 1.0 + sigma_down ** 1.0) / 2) ** 1.0
    return sigma_up, sigma_down, sigma_mid


def _make_decoder_deterministic(path):
    """Return decoder model bytes with the two random nodes (RandomNormalLike
    for the noise excitation, RandomUniformLike for the harmonic initial phase)
    replaced by deterministic zeros (Sub(x, x)). onnxruntime advances its RNG
    across run() calls even with a fixed seed attribute, so removing the random
    ops is the reliable way to get bit-exact, reproducible output."""
    m = onnx.load(path)
    for n in m.graph.node:
        if n.op_type in ("RandomNormalLike", "RandomUniformLike"):
            xin = n.input[0]
            yout = n.output[0]
            del n.attribute[:]
            n.op_type = "Sub"
            del n.input[:]
            n.input.extend([xin, xin])
            del n.output[:]
            n.output.extend([yout])
    return m.SerializeToString()


class StyleTTS2ONNX:
    def __init__(self, model_dir, providers=None, deterministic=False, seed=0):
        with open(os.path.join(model_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        if providers is None:
            avail = ort.get_available_providers()
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"])
        so = ort.SessionOptions()
        self.deterministic = deterministic
        self.default_seed = seed

        def sess(name):
            return ort.InferenceSession(os.path.join(model_dir, name), so, providers=providers)

        self.encoder = sess("encoder.onnx")
        self.diffusion = sess("diffusion.onnx")
        self.predictor = sess("predictor.onnx")
        if deterministic:
            self.decoder = ort.InferenceSession(
                _make_decoder_deterministic(os.path.join(model_dir, "decoder.onnx")),
                so, providers=providers)
        else:
            self.decoder = sess("decoder.onnx")
        self.tokenizer = TextCleaner()
        self.style_dim = self.meta["style_dim"]
        self.noise_dim = self.meta["noise_dim"]

    # ---- diffusion ---------------------------------------------------------
    def _denoise(self, x, sigma, bert_dur, scale):
        out = self.diffusion.run(["denoised"], {
            "x": x.astype(np.float32),
            "sigma": np.array([sigma], dtype=np.float32),
            "bert_dur": bert_dur.astype(np.float32),
            "embedding_scale": np.array([scale], dtype=np.float32),
        })[0]
        return out

    def _sample_style(self, bert_dur, num_steps, embedding_scale, rng):
        d = self.meta["diffusion"]
        sigmas = karras_schedule(num_steps, d["sigma_min"], d["sigma_max"], d["rho"])
        noise = rng.standard_normal((1, 1, self.noise_dim)).astype(np.float32)
        x = sigmas[0] * noise
        for i in range(num_steps - 1):
            sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
            sigma_up, sigma_down, sigma_mid = _adpm2_sigmas(sigma, sigma_next)
            d0 = (x - self._denoise(x, sigma, bert_dur, embedding_scale)) / sigma
            x_mid = x + d0 * (sigma_mid - sigma)
            d_mid = (x_mid - self._denoise(x_mid, sigma_mid, bert_dur, embedding_scale)) / sigma_mid
            x = x + d_mid * (sigma_down - sigma)
            x = x + rng.standard_normal(x.shape).astype(np.float32) * sigma_up
        return x.reshape(1, self.noise_dim)          # s_pred

    # ---- tokens ------------------------------------------------------------
    def _tokens(self, phonemes):
        t = self.tokenizer(phonemes)
        t.insert(0, 0)
        return t

    # ---- full pipeline -----------------------------------------------------
    def inference(self, phonemes, additional_ph=None, speed=1.0,
                  diffusion_steps=5, embedding_scale=1.0, s_prev=None,
                  alpha=0.7, seed=None):
        """Single-speaker inference matching inference2() + the workaround.

        With deterministic=True (set on the constructor) and a fixed seed, the
        output is bit-exact reproducible across runs."""
        if seed is None:
            seed = self.default_seed
        rng = np.random.default_rng(seed)

        tokens_orig = self._tokens(phonemes.strip())
        if additional_ph:
            tokens_tph = self._tokens(additional_ph.strip())
            # additional text is appended at the END for single-speaker models
            tokens = tokens_orig + self.tokenizer(" ") + tokens_tph
        else:
            tokens_tph = None
            tokens = tokens_orig

        tokens_np = np.array([tokens], dtype=np.int64)

        bert_dur, d_en, t_en = self.encoder.run(
            ["bert_dur", "d_en", "t_en"], {"tokens": tokens_np})

        s_pred = self._sample_style(bert_dur, diffusion_steps, embedding_scale, rng)
        if s_prev is not None:
            s_pred = alpha * s_prev + (1 - alpha) * s_pred

        s = s_pred[:, self.style_dim:]        # prosody style (1,128)
        ref = s_pred[:, :self.style_dim]      # acoustic style (1,128)

        d, duration = self.predictor.run(["d", "duration"],
                                         {"d_en": d_en, "s": s.astype(np.float32)})

        # ---- durations + short-sentence workaround (see inference2) --------
        duration = duration[0] / speed                       # (N,)
        pred_dur = np.clip(np.round(duration), 1, None).astype(np.int64)
        pred_dur[-1] = min(int(pred_dur[-1]), 6)

        if tokens_tph is not None:
            # remove the appended filler: zero the space + the filler tokens
            orig_len = len(tokens_orig)
            pred_dur[orig_len] = 0                            # the joining space
            for i in range(len(tokens_tph)):
                if tokens_tph[i] == tokens[i + orig_len + 1]:
                    pred_dur[i + orig_len + 1] = 0

        total = int(pred_dur.sum())
        n = pred_dur.shape[0]
        aln = np.zeros((1, n, total), dtype=np.float32)
        c = 0
        for i in range(n):
            aln[0, i, c:c + int(pred_dur[i])] = 1
            c += int(pred_dur[i])

        audio = self.decoder.run(["audio"], {
            "d": d, "t_en": t_en, "s": s.astype(np.float32),
            "ref": ref.astype(np.float32), "aln": aln})[0]
        return audio[0], s_pred


if __name__ == "__main__":
    import soundfile as sf

    p = argparse.ArgumentParser("Run exported StyleTTS2 ONNX model")
    p.add_argument("-m", required=True, help="directory with the exported onnx graphs")
    p.add_argument("-t", required=True, help="text to speak (already plain text)")
    p.add_argument("-o", default="onnx_out.wav")
    p.add_argument("-s", type=float, default=1.0, help="speech speed")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true",
                   help="bit-exact reproducible output (zeros the decoder noise excitation)")
    p.add_argument("--filler", default="That's what we have for now.",
                   help="filler sentence appended then removed for short inputs")
    p.add_argument("--no-filler", action="store_true")
    args = p.parse_args()

    import phonemizer
    gp = phonemizer.backend.EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
    from nltk.tokenize import word_tokenize

    def phon(text):
        return " ".join(word_tokenize(gp.phonemize([text])[0]))

    model = StyleTTS2ONNX(args.m, deterministic=args.deterministic, seed=args.seed)

    ph = phon(args.t)
    add_ph = None
    if not args.no_filler:
        words = [w for w in word_tokenize(args.t) if w.isalnum()]
        if len(words) < 6:
            add_ph = phon(args.filler)

    audio, _ = model.inference(ph, additional_ph=add_ph, speed=args.s,
                               diffusion_steps=args.steps, embedding_scale=args.scale,
                               seed=args.seed)
    sf.write(args.o, audio, model.meta["sample_rate"])
    print("wrote", args.o, "samples:", audio.shape)
