"""
Diagnose whether the StyleTTS2 text aligner needs retraining for your data.

Reproduces the exact alignment path used in train_first.py so you can grade the
aligner on YOUR phonemes (with the inline tone arrows) without the old log.

Usage:
    # stock aligner (never saw tone arrows):
    python diagnose_aligner.py --val_list Data/val_list.txt --root_path Data/wavs \
        --aligner Utils/ASR/epoch_00080.pth --asr_config Utils/ASR/config.yml \
        --out_dir diag_stock

    # your stage-1 fine-tuned aligner (extracted from epoch_1st_*.pth):
    python diagnose_aligner.py --val_list Data/val_list.txt --root_path Data/wavs \
        --stage1_ckpt Models/epoch_1st_00XX.pth --asr_config Utils/ASR/config.yml \
        --out_dir diag_finetuned

    # control: stock aligner on the SAME audio but with tone arrows removed
    python diagnose_aligner.py --val_list Data/val_list.txt --root_path Data/wavs \
        --aligner Utils/ASR/epoch_00080.pth --asr_config Utils/ASR/config.yml \
        --strip_arrows --out_dir diag_stock_noarrows

Use Utils/extract_text_aligner.py to pull a standalone text_aligner .pth out of
an epoch_1st_*.pth if you'd rather load it via --aligner.

Read the printed summary + the saved attention PNGs. See the interpretation
notes at the bottom of this file.
"""
import os, argparse, collections
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import load_ASR_models
from meldataset import build_dataloader
from utils import maximum_path, mask_from_lens, length_to_mask
from text_utils import symbols  # index -> symbol, to label arrows/vowels

TONE_ARROWS = set("→↗↓↘")
VOWELS = set("aeiouɑɐɒæɔəɚɛɜɤʊʌyɪ")   # rough IPA vowel set for bucketing
ARROW_IDS = {i for i, s in enumerate(symbols) if s in TONE_ARROWS}


def strip_arrow_tokens(texts, input_lengths, pad_id=0):
    """Remove tone-arrow tokens from a padded (B, T) text batch, left-shifting
    survivors and decrementing each row's length. The mel/audio side is left
    untouched, so this is a controlled "same audio, tone-free phonemes" probe:
    if a stock aligner aligns markedly better here than with arrows present,
    the arrows are out-of-distribution for it."""
    B = texts.size(0)
    new_lengths = input_lengths.clone()
    rows = []
    for b in range(B):
        L = int(input_lengths[b])
        keep = [texts[b, t] for t in range(L) if int(texts[b, t]) not in ARROW_IDS]
        new_lengths[b] = len(keep)
        rows.append(keep)
    Tmax = int(new_lengths.max())
    new_texts = torch.full((B, Tmax), pad_id, dtype=texts.dtype, device=texts.device)
    for b, keep in enumerate(rows):
        for j, v in enumerate(keep):
            new_texts[b, j] = v
    return new_texts, new_lengths


def load_aligner(args, device):
    aligner = load_ASR_models(args.aligner or "Utils/ASR/epoch_00080.pth", args.asr_config)
    if args.stage1_ckpt:
        sd = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
        params = sd["net"] if "net" in sd else sd
        ta = params["text_aligner"]
        ta = {k[7:] if k.startswith("module.") else k: v for k, v in ta.items()}
        aligner.load_state_dict(ta)
        print(f"Loaded fine-tuned text_aligner from {args.stage1_ckpt}")
    return aligner.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_list", required=True)
    ap.add_argument("--root_path", default="")
    ap.add_argument("--asr_config", default="Utils/ASR/config.yml")
    ap.add_argument("--aligner", default=None, help="stock/plain aligner .pth")
    ap.add_argument("--stage1_ckpt", default=None, help="epoch_1st_*.pth to pull text_aligner from")
    ap.add_argument("--out_dir", default="diag_aligner")
    ap.add_argument("--n_batches", type=int, default=50)
    ap.add_argument("--n_plots", type=int, default=20)
    ap.add_argument("--strip_arrows", action="store_true",
                    help="control run: drop tone-arrow tokens so the aligner sees "
                         "tone-free phonemes on the same audio (isolates whether "
                         "the arrows are what breaks alignment)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    aligner = load_aligner(args, device)
    n_down = aligner.n_down

    with open(args.val_list, encoding="utf-8") as f:
        val_list = f.readlines()

    loader = build_dataloader(val_list, args.root_path, validation=True,
                              batch_size=4, num_workers=2, device=device,
                              dataset_config={}, dynamic_batch=False)

    mono_losses, s2s_accs = [], []
    dur_by_bucket = collections.defaultdict(list)     # 'arrow'/'vowel'/'other' -> durations
    recall_hit = collections.Counter(); recall_tot = collections.Counter()
    plotted = 0

    for bi, batch in enumerate(loader):
        if bi >= args.n_batches:
            break
        # batch = [waves, texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ...]
        texts, input_lengths, _, _, mels, mel_input_length, _ = [
            b.to(device) if torch.is_tensor(b) else b for b in batch[1:8]]

        if args.strip_arrows:
            texts, input_lengths = strip_arrow_tokens(texts, input_lengths)

        with torch.no_grad():
            mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
            text_mask = length_to_mask(input_lengths).to(device)
            ppgs, s2s_pred, s2s_attn = aligner(mels, mask, texts)

            s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
            attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
            attn_mask = attn_mask * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
            attn_mask = (attn_mask < 1)
            s2s_attn = s2s_attn.masked_fill(attn_mask, 0.0)

            mask_ST = mask_from_lens(s2s_attn.float(), input_lengths, mel_input_length // (2 ** n_down))
            s2s_attn_mono = maximum_path(s2s_attn.float(), mask_ST)

            # (1) mono agreement — the key number stage-1 minimizes (x10 as in train)
            mono_losses.append((torch.nn.functional.l1_loss(s2s_attn, s2s_attn_mono) * 10).item())

            # (2) per-phoneme durations from the monotonic path + per-symbol s2s recall
            durs = s2s_attn_mono.sum(dim=-1)  # (B, T_text)
            pred = s2s_pred.argmax(-1)        # (B, T_text+1)
            for b in range(texts.size(0)):
                L = int(input_lengths[b])
                for t in range(L):
                    sym = symbols[int(texts[b, t])]
                    bucket = "arrow" if sym in TONE_ARROWS else ("vowel" if sym in VOWELS else "other")
                    dur_by_bucket[bucket].append(float(durs[b, t]))
                    recall_tot[bucket] += 1
                    if int(pred[b, t]) == int(texts[b, t]):
                        recall_hit[bucket] += 1
                s2s_accs.append(float((pred[b, 1:L+1] == texts[b, :L]).float().mean()))

            # (3) save a few attention heatmaps with arrow rows highlighted
            for b in range(texts.size(0)):
                if plotted >= args.n_plots:
                    break
                L = int(input_lengths[b]); Tm = int(mel_input_length[b] // (2 ** n_down))
                A = s2s_attn[b, :L, :Tm].cpu().numpy()
                fig, ax = plt.subplots(figsize=(min(20, Tm/20 + 3), min(30, L/5 + 2)))
                ax.imshow(A, aspect="auto", origin="lower", interpolation="nearest")
                labels = [symbols[int(texts[b, t])] for t in range(L)]
                ax.set_yticks(range(L)); ax.set_yticklabels(labels, fontsize=5)
                for t, s in enumerate(labels):
                    if s in TONE_ARROWS:
                        ax.axhline(t, color="red", lw=0.3, alpha=0.5)
                ax.set_xlabel("mel frames /2"); ax.set_ylabel("phoneme (arrows = red)")
                fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, f"attn_{plotted:03d}.png"), dpi=120)
                plt.close(fig); plotted += 1

    print("\n==== ALIGNER DIAGNOSTIC ====")
    print(f"batches={len(mono_losses)}  plots -> {args.out_dir}/attn_*.png")
    print(f"mono L1 loss (x10)  mean={np.mean(mono_losses):.3f}  (lower=better, <~1 healthy)")
    print(f"s2s phoneme acc     mean={np.mean(s2s_accs):.3f}  (higher=better)")
    print("\nper-symbol-class s2s recall (arrows lagging = tones are OOD for the aligner):")
    for k in ("vowel", "arrow", "other"):
        if recall_tot[k]:
            print(f"  {k:6s} recall={recall_hit[k]/recall_tot[k]:.3f}  n={recall_tot[k]}")
    print("\nper-symbol-class monotonic-path duration (frames/2):")
    for k in ("vowel", "arrow", "other"):
        if dur_by_bucket[k]:
            d = np.array(dur_by_bucket[k])
            print(f"  {k:6s} mean={d.mean():.2f}  std={d.std():.2f}  p95={np.percentile(d,95):.1f}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# HOW TO READ THE OUTPUT
# ---------------------------------------------------------------------------
# Run it TWICE: once with --aligner Utils/ASR/epoch_00080.pth (stock, never saw
# arrows) and once with --stage1_ckpt <your epoch_1st_*.pth> (your fine-tuned).
#
# 1) attn_*.png  — want a crisp near-diagonal staircase. Red lines mark the tone
#    arrows. GOOD: arrow rows are thin, sit on/next to their vowel. BAD: arrow
#    rows are wide/blurry or steal a big block of frames from the vowel -> the
#    exact OOD-tone symptom that flattens tone contours.
#
# 2) mono L1 loss (x10) — the number stage-1 actually minimized. If your
#    fine-tuned aligner is still high here (well above the stock aligner's value
#    on clean English), fine-tuning never absorbed the arrows.
#
# 3) per-class s2s recall — if 'vowel'/'other' recall is high but 'arrow' recall
#    is much lower, the aligner can't recognize tones from audio -> retrain.
#
# 4) per-class duration — arrows should be ~0 or small & consistent. Huge mean
#    or std for 'arrow' means the duration TARGETS your whole model learned from
#    are noisy -> retrain the aligner (AuxiliaryASR) on your exact G2P.
#
# 5) --strip_arrows CONTROL — run the SAME (usually stock) aligner twice, once
#    normally and once with --strip_arrows. If mono loss drops a lot / matrices
#    sharpen once arrows are removed, the arrows are what the aligner chokes on
#    (they are OOD for it). If alignment is already clean WITH arrows, tones are
#    not the problem and retraining the aligner for tones won't help.
#
# DECISION:
#   fine-tuned aligner: clean matrices + low mono loss + arrows ~0/consistent
#       -> aligner is FINE, prosody problem is elsewhere (PL-BERT / segmentation)
#   fine-tuned aligner: still blurry / high mono loss / bad arrow stats,
#       AND --strip_arrows on the stock aligner is markedly cleaner
#       -> RETRAIN the text aligner on Chinese with your arrow-tone G2P
