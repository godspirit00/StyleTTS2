"""
Extract a standalone text-aligner checkpoint from a StyleTTS2 stage-1 model.

A stage-1 checkpoint (epoch_1st_*.pth) bundles every sub-model under
params["net"]["text_aligner"] (often with "module." prefixes from DDP). This
pulls just the fine-tuned text_aligner out and writes it in the same format
load_ASR_models() expects -- a dict with a top-level "model" key -- so you can
load it anywhere the stock Utils/ASR/epoch_00080.pth is used, e.g.:

    python diagnose_aligner.py --val_list Data/val_list.txt \
        --aligner Utils/ASR/text_aligner_finetuned.pth ...

Usage:
    python Utils/extract_text_aligner.py Models/epoch_1st_00XX.pth \
        -o Utils/ASR/text_aligner_finetuned.pth

    # sanity-check it actually loads against the ASR config:
    python Utils/extract_text_aligner.py Models/epoch_1st_00XX.pth \
        -o Utils/ASR/text_aligner_finetuned.pth --verify Utils/ASR/config.yml
"""
import os
import argparse
from collections import OrderedDict

import torch


def extract(stage1_path, out_path):
    ckpt = torch.load(stage1_path, map_location="cpu", weights_only=False)
    params = ckpt["net"] if "net" in ckpt else ckpt
    if "text_aligner" not in params:
        raise KeyError(
            f"'text_aligner' not found in {stage1_path}. "
            f"Top-level keys present: {list(params.keys())}")

    raw = params["text_aligner"]
    state = OrderedDict()
    for k, v in raw.items():
        state[k[7:] if k.startswith("module.") else k] = v

    torch.save({"model": state}, out_path)
    print(f"Wrote {out_path}  ({len(state)} tensors)")
    return out_path


def verify(out_path, asr_config):
    # imported lazily so plain extraction has no dependency on the model code
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import load_ASR_models
    model = load_ASR_models(out_path, asr_config)
    n = sum(p.numel() for p in model.parameters())
    print(f"Verify OK: loaded into ASRCNN via load_ASR_models  ({n:,} params)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage1_ckpt", help="path to epoch_1st_*.pth")
    ap.add_argument("-o", "--out", required=True, help="output .pth path")
    ap.add_argument("--verify", metavar="ASR_CONFIG", default=None,
                    help="also load the result back via load_ASR_models to confirm "
                         "it matches (pass Utils/ASR/config.yml)")
    args = ap.parse_args()

    out = extract(args.stage1_ckpt, args.out)
    if args.verify:
        verify(out, args.verify)


if __name__ == "__main__":
    main()
