"""
Build a StyleTTS2 train_list from the BiaoBei (标贝 / BZNSYP) dataset, using the
corpus's own prosody labels (#1 #2 #3 #4) for prosodic-word grouping.

Input: the ProsodyLabeling file (000001-010000.txt / "prosodylabeling.txt"),
two lines per utterance:

    000001\t卡尔普#2陪外孙#1玩滑梯#4。
    \tka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1

Output (StyleTTS2 format):  <wav>|<phonemes>|<speaker_id>

    000001.wav|kʰˈa↗ˈɤ↗ɹpʰˈu↓ pʰˈe↗ɪwˈa↘ɪsˈwə→n wˈa↗nxˈwa↗tʰˈi→ .|0

Why the grouping
----------------
'#n' marks are prosodic boundaries: #1 prosodic word, #2 prosodic phrase,
#3 intonation phrase, #4 sentence. Syllables inside one '#'-group form a
prosodic word, so we write them as ONE contiguous phoneme run and put a space
only at #1/#2 boundaries -- mirroring how StyleTTS2 writes every other language
(space = word boundary, contiguous phonemes within a word). #3/#4 essentially
always coincide with the sentence's own punctuation (，。！？、), which we emit
as the pause token exactly as the original per-syllable script did; a bare #3
with no punctuation just yields a word-boundary space.

This reuses YOUR pinyin->IPA converter and tone mapping, so the segmental output
is byte-identical to your existing pipeline -- only the spacing/grouping changes.

Usage
-----
    python preprocess_biaobei.py \
        --labeling  BZNSYP/ProsodyLabeling/000001-010000.txt \
        --table     BZNSYP/pinyin2ipa.txt \
        --wav_dir   Wave \
        --out       Data/biaobei_list.txt \
        --speaker_id 0
        # --frontend_dir <dir containing pinyin_ipa_converter.py>  (if not importable)
"""
import os
import sys
import argparse

# ---- tone / punctuation handling, kept identical to your hanzi_to_ipa.py ----

def retone(p: str) -> str:
    p = p.replace('˧˩˧', '↓')   # third tone
    p = p.replace('˧˥', '↗')    # second tone
    p = p.replace('˥˩', '↘')    # fourth tone
    p = p.replace('˥', '→')     # first tone
    p = p.replace(chr(635) + chr(809), 'ɨ').replace(chr(633) + chr(809), 'ɨ')
    assert chr(809) not in p, p
    return p


# hanzi punctuation -> pause token (as your map_punctuation produces)
PUNCT_MAP = {
    '、': ',', '，': ',', '。': '.', '．': '.', '！': '!', '：': ':',
    '；': ';', '？': '?', '«': '“', '»': '”', '《': '“', '》': '”',
    '「': '“', '」': '”', '【': '“', '】': '”', '（': '“', '）': '”',
    '—': '—', '…': '…',
}


def is_cjk(ch: str) -> bool:
    return '一' <= ch <= '鿿' or ch in ('儿',)


def convert_syllable(py, converter, fails):
    ipa = converter.convert(py)
    if ipa is None:
        fails.append(py)
        return None
    return retone(ipa)


def build_phonemes(hanzi_line, pinyin_line, converter, fails):
    """Walk the hanzi (with #markers/punctuation) alongside the pinyin syllables,
    emitting contiguous prosodic words separated by spaces, with punctuation as
    pause tokens."""
    pinyins = pinyin_line.split()
    pi = 0
    parts = []          # finished tokens (words / pauses)
    word = []           # phonemes of the current prosodic word (contiguous)

    def flush_word():
        if word:
            parts.append(''.join(word))
            word.clear()

    i = 0
    n = len(hanzi_line)
    while i < n:
        ch = hanzi_line[i]
        if ch == '#':
            # '#' + single digit boundary marker -> word boundary (space)
            flush_word()
            i += 2  # skip '#' and its level digit
            continue
        if is_cjk(ch):
            if pi < len(pinyins):
                ipa = convert_syllable(pinyins[pi], converter, fails)
                pi += 1
                if ipa is not None:
                    word.append(ipa)
            i += 1
            continue
        if ch in PUNCT_MAP:
            flush_word()
            parts.append(PUNCT_MAP[ch])
            i += 1
            continue
        # anything else (spaces, stray chars) -> ignore
        i += 1

    flush_word()
    # join words/pauses with single spaces; this yields "word word . " style output
    return ' '.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeling", required=True, help="ProsodyLabeling txt file")
    ap.add_argument("--table", required=True, help="pinyin2ipa.txt")
    ap.add_argument("--wav_dir", default="", help="prefix for the .wav path column")
    ap.add_argument("--wav_ext", default=".wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--speaker_id", default="0")
    ap.add_argument("--frontend_dir", default=None,
                    help="dir containing pinyin_ipa_converter.py, if not importable")
    args = ap.parse_args()

    if args.frontend_dir:
        sys.path.insert(0, args.frontend_dir)
    from pinyin_ipa_converter import load_converter
    converter = load_converter(args.table)

    # read the two-line-per-utterance format
    with open(args.labeling, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    out_rows = []
    fails = []
    n_utt = 0
    i = 0
    while i < len(lines):
        head = lines[i]
        if "\t" not in head:
            i += 1
            continue
        uid, hanzi = head.split("\t", 1)
        pinyin = lines[i + 1].strip() if i + 1 < len(lines) else ""
        i += 2
        n_utt += 1
        phonemes = build_phonemes(hanzi, pinyin, converter, fails)
        wav = os.path.join(args.wav_dir, uid + args.wav_ext) if args.wav_dir else uid + args.wav_ext
        out_rows.append(f"{wav}|{phonemes}|{args.speaker_id}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out_rows) + "\n")

    print(f"Wrote {len(out_rows)} / {n_utt} utterances -> {args.out}")
    if fails:
        from collections import Counter
        c = Counter(fails)
        print(f"WARNING: {len(fails)} syllables had no table entry (dropped). Top:")
        for py, k in c.most_common(20):
            print(f"    {py}  x{k}")
        print("Add these to pinyin2ipa.txt (or check tone-digit format) and re-run.")


if __name__ == "__main__":
    main()
