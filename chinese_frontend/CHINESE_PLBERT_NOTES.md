# Training a Chinese PL-BERT for this StyleTTS2 (handoff note)

Carry this into the PL-BERT training session. It records the decisions and the
hard compatibility constraints so the resulting checkpoint drops into this repo
and stays consistent with the frontend fixes in `chinese_frontend/`.

Reference repo: https://github.com/yl4579/PL-BERT (the `train.ipynb` /
`preprocess.ipynb` recipe). Target arch is `Utils/PLBERT/config.yml` in this repo.

---

## 0. First, decide if you even need it

The frontend fixes (table corrections + prosodic-word grouping) change the token
stream the acoustic model AND the PL-BERT see. Retrain StyleTTS2 with those +
the existing multilingual PL-BERT first and listen. If prosody is acceptable,
you may not need a dedicated PL-BERT. Train the Chinese PL-BERT only if the
multilingual one is still the bottleneck. (PL-BERT is cheap, but not free.)

---

## 1. Hard compatibility constraints (do not deviate)

These are what make the checkpoint loadable here. StyleTTS2 loads only the
Albert **body** (`CustomAlbert`) and feeds it `TextCleaner` phoneme IDs directly
(see `styletts2_model_loader`: `self.model.bert(tokens_tensors, ...)`). The
grapheme-prediction head from PL-BERT training is discarded.

1. **Phoneme input vocab = this repo's `TextCleaner.symbols`, same IDs.**
   - `len(symbols) == 178`, space `" "` is ID 16.
   - The PL-BERT phoneme token IDs during pretraining MUST equal these IDs, i.e.
     tokenize phoneme strings with `TextCleaner` (char-by-char), not with a
     freshly-built vocab. If you use the PL-BERT repo's `token_maps.pkl`, make
     that map the identity over `TextCleaner.dicts`.
   - Consequence: `vocab_size: 178` in the model config.

2. **Architecture must equal `Utils/PLBERT/config.yml`:**
   `hidden_size: 768`, `num_hidden_layers: 12`, `num_attention_heads: 12`,
   `intermediate_size: 2048`, `max_position_embeddings: 512`, `dropout: 0.1`,
   `vocab_size: 178`. Keep `max_position_embeddings: 512`; sequences longer than
   that must be split.

3. **Phonemization = the exact `chinese_frontend` frontend.** Same corrected
   `pinyin2ipa.txt`, same tone arrows, same jieba prosodic-word grouping
   (contiguous phonemes within a word, space between words). If PL-BERT is
   pretrained on a different spacing/symbol convention than the acoustic model
   uses at inference, its features are out-of-distribution — the whole reason
   the multilingual one underperforms.

---

## 2. Corpus

- Chinese Wikipedia dump + a cleaner news/subtitle corpus (Wikipedia alone is
  encyclopedic and stiff; mixing in conversational text helps prosody).
- Normalize numbers/dates/symbols to spoken form FIRST (WeTextProcessing /
  `tn.chinese.normalizer`, the same normalizer the inference path uses), then
  phonemize. Text→spoken normalization mismatches are a common silent quality
  drain.
- Segment each sentence with jieba, phonemize with the grouped frontend. Store,
  per sentence: the phoneme-ID sequence (TextCleaner IDs) and the grapheme
  target sequence (see below).

---

## 3. The one real design decision: grapheme-prediction target

PL-BERT's pretext = masked phoneme modeling **plus** predicting the grapheme of
each phoneme. Because the head is discarded for StyleTTS2, this choice only
affects pretraining signal quality, not compatibility — so optimize for signal.

Recommended (keeps prosodic grouping AND rich char-level semantics):

- **Word unit for spacing/masking = jieba prosodic word** (matches the acoustic
  stream). `word_mask_prob` masks a whole prosodic word at a time.
- **Per-phoneme grapheme target = the Han character of that phoneme's syllable**
  (each Mandarin syllable = one character), via a **char-level** tokenizer such
  as `bert-base-chinese`. So every phoneme of syllable 卡 targets the id of 卡.
  This requires a small change to the PL-BERT label builder: assign per-syllable
  char targets instead of one word-level target per word.

Simpler fallback (closer to the stock repo, weaker grouping): treat each Han
character as its own "word" — but that reintroduces the every-syllable-is-a-word
spacing in the phoneme stream, which is exactly what we removed. Avoid unless the
per-syllable label change above is too fiddly.

Keep masking/replace probs at the repo defaults to start:
`word_mask_prob: 0.15`, `phoneme_mask_prob: 0.1`, `replace_prob: 0.2`,
`token_mask: "M"` (ensure "M" maps to a reserved phoneme ID that isn't a real
symbol; the stock config uses a mask token — verify it doesn't collide with the
178 real symbols).

---

## 4. Training

- Follow `PL-BERT/train.ipynb`. Loss = masked-phoneme CE + grapheme CE.
- Steps: stock English used 1M @ batch 192. For a Chinese corpus, 500k–1M steps
  is a reasonable range; watch both losses plateau. Single 3090/4090, fp16:
  order of several days to ~2 weeks depending on corpus size and steps.
- Save in the format this repo expects: the loader reads a `.t7`/`.pth` and a
  sibling `config.yml` from `PLBERT_dir` (or `plbert_params` inline in the
  StyleTTS2 config). Mirror `Utils/PLBERT/`'s layout (`config.yml` +
  `step_XXXX.t7`) and point `PLBERT_dir` at it.

---

## 5. Wiring it into StyleTTS2

- Put the new checkpoint + its `config.yml` in a dir; set `PLBERT_dir` in the
  StyleTTS2 training config to it (the loader falls back to `Utils/PLBERT` if
  unset — don't rely on that).
- The `plbert_encoder = Linear(hidden_size, hidden_dim)` is learned fresh in
  StyleTTS2 training, so hidden_size just has to be consistent (768).
- Retrain StyleTTS2 stage 1 + 2 from scratch with: corrected table, jieba
  grouping, BiaoBei prosody preprocessing, and this PL-BERT. All four must agree
  on the phoneme convention.

---

## 6. Consistency checklist (the failure modes)

- [ ] PL-BERT phoneme IDs == `TextCleaner` IDs (178, space=16).
- [ ] Same corrected `pinyin2ipa.txt` everywhere.
- [ ] Same jieba grouping in: PL-BERT corpus, BiaoBei training list, inference.
- [ ] Same text normalizer in PL-BERT corpus prep and inference.
- [ ] `max_position_embeddings` 512 respected (split long sentences).
- [ ] Mask token ID reserved, not colliding with a real symbol.
- [ ] StyleTTS2 retrained from scratch (new token stream ≠ old checkpoints).
