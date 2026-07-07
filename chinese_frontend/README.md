# Chinese frontend fixes for StyleTTS2

These are drop-in helpers for a Chinese StyleTTS2 pipeline. They fix two prosody
issues in the pinyin→IPA frontend:

1. **Table bugs** — a few `pinyin2ipa.txt` rows inject noise (a vowel-less
   syllable, mis-copied erhua rows).
2. **No word grouping** — the original processing space-separates *every*
   syllable, so the duration predictor and PL-BERT see each syllable as its own
   word. StyleTTS2 represents every other language as *space = word boundary,
   phonemes contiguous within a word*; these make Chinese match that.

## Files

| File | What it does |
|------|--------------|
| `fix_pinyin2ipa_table.py` | Rewrites the known-bad rows in your `pinyin2ipa.txt`, leaving everything else untouched. Run once. |
| `preprocess_biaobei.py` | Builds a StyleTTS2 `train_list` from BiaoBei/BZNSYP, using the corpus's `#1 #2 #3 #4` prosody labels for prosodic-word grouping. |
| `hanzi_to_ipa_grouped.py` | Inference-time `hanzi_to_ipa()` that groups syllables into words with `jieba`, matching the training convention. |
| `CHINESE_PLBERT_NOTES.md` | Handoff note for training a drop-in Chinese PL-BERT. |
| `DECISIONS.md` | Running record of what we changed and what we deliberately left alone (aligner, SLM/WavLM) and why. |

## Workflow

```bash
# 1. fix the conversion table
python fix_pinyin2ipa_table.py BZNSYP/pinyin2ipa.txt -o pinyin2ipa.txt

# 2. build the training list from BiaoBei prosody labels
python preprocess_biaobei.py \
    --labeling BZNSYP/ProsodyLabeling/000001-010000.txt \
    --table    pinyin2ipa.txt \
    --wav_dir  Wave \
    --out      Data/biaobei_list.txt \
    --speaker_id 0

# 3. at inference, use the grouped frontend (place pinyin2ipa.txt next to it)
#    import hanzi_to_ipa from hanzi_to_ipa_grouped instead of the old module
```

The segmental IPA + tone-arrow output is **identical** to your existing
pipeline; only the spacing/grouping changes. So the phoneme symbol set is
unchanged and `TextCleaner` needs no new tokens.

## Consistency note

Training uses BiaoBei's gold `#1/#2` boundaries; inference uses `jieba` lexical
words as an approximation of prosodic words. They won't match exactly, but both
give real grouping. When you later train a dedicated Chinese PL-BERT, phonemize
its corpus with this same grouping (jieba) so the whole stack agrees.
