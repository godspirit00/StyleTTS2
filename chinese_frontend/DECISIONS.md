# Chinese StyleTTS2 — decisions & rationale

Running record of the "do I need to change X?" questions, so they aren't
re-litigated later. Priorities: the prosody problem is driven by the **frontend
/ PL-BERT / grouping**, not by the aligner or the SLM.

## Text aligner — NOT retraining (diagnosed healthy)

Ran `diagnose_aligner.py` on the fine-tuned stage-1 aligner vs. the stock one.
The fine-tuned aligner handles the tone arrows well:

- mono L1 (×10) = 0.057 (crisp monotonic), vs 0.124 stock
- per-class recall: vowel 0.992, **arrow 0.982**, other 0.991
  (stock scored **0.000** on arrows — fine-tuning successfully absorbed them)
- tone-arrow durations tidy (~1.7 frames, low variance)

Conclusion: the aligner is not the bottleneck; do not retrain it. (Note: the
aggregate "s2s phoneme acc" line had an off-by-one, since fixed — trust the
per-class recall.)

## SLM / WavLM — NOT replacing (wrong lever for prosody)

Ref: yl4579/StyleTTS2 discussions/111 suggests Whisper/multilingual-wav2vec2 for
Chinese; but Bert-VITS2 uses `wavlm-base-plus` for Chinese successfully.

- The SLM (`microsoft/wavlm-base-plus`, `Configs/config.yml:116`) is a **frozen
  feature extractor** used only in `WavLMLoss` (feature-matching + a small
  adversarial head) during stage-2 joint training. It's an audio-naturalness /
  anti-artifact signal, **not** a prosody/rhythm signal — so swapping it cannot
  fix flat intonation or choppy rhythm.
- Self-supervised speech encoders learn largely **language-agnostic** acoustic
  features; WavLM doesn't need to "know" Mandarin to spot vocoder artifacts.
  Bert-VITS2 borrowed this exact SLM-adversarial idea and uses WavLM for Chinese.
- The discussion's "WavLM only supports English" is imprecise (its pretraining
  includes multilingual VoxPopuli), and the advice there is "you can try," not a
  necessity.

Conclusion: leave WavLM as-is. If you ever want to A/B it as a late polish:

| Candidate | Effort | Notes |
|-----------|--------|-------|
| `TencentGameMate/chinese-hubert-base` | **drop-in** | Same Wav2Vec2-style API, 768-dim / 12-layer → matches `hidden: 768`, `nlayers: 13`; only the model string changes. |
| `facebook/wav2vec2-large-xlsr-53` | config change | Multilingual, API-compatible, but 1024-dim / 24-layer → update `hidden`, `nlayers`, and the `wd` head channels. |
| Whisper encoder | code change | Takes log-mel, not raw waveform; not a plain `AutoModel(input_values=...)`. |

Only worth trying after the frontend + PL-BERT work; expect a second-order
fidelity effect at most.
