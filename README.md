# Assignment 4: RVQ-Based NAC Analysis — Claude Code Prompt

## Task Summary

Analyze performance degradation in RVQ-based Neural Audio Codec (NAC) by encoding/decoding audio at varying RVQ levels and measuring reconstruction quality via three metrics.

---

## Implementation Requirements

### Model
- Use **EnCodec** or **DAC** (24 kHz checkpoint)
- Resample all input audio to **24 kHz** before codec inference

### Datasets
- **LJSpeech**: validation split (from previous assignment)
- **VCTK**: speakers **p232** and **p257** only (test split)

### RVQ Level Configurations
Run encode->decode for each setting:
- Full RVQ levels (all available, e.g. 8 or 12 depending on model)
- 5 RVQ levels
- 3 RVQ levels
- 2 RVQ levels
- 1 RVQ level

---

## Metrics to Compute

### 1. LogMel Difference (ℓ₁) ↓
```
D_mel = (1 / T*F) * || log MelSpec(x) - log MelSpec(x_hat) ||_1
```
MelSpectrogram config:
- Sample rate: 24,000 Hz
- FFT size: 1024
- Window length: 1024
- Hop length: 256
- Mel bins: 80
- Frequency range: 0–12,000 Hz

For "Original wav" row: LogMel diff is `-` (N/A, reference itself).

### 2. WER ↓
- Use a pretrained ASR model (e.g. `openai/whisper-base` or `facebook/wav2vec2-base-960h`) to transcribe both original and reconstructed audio
- Compute WER between transcription of original and transcription of reconstructed
- For "Original wav" row: run ASR on original and compute WER against ground-truth transcripts if available, or report 0/baseline

### 3. UTMOS ↑
- Use **UTMOS** (https://github.com/sarulab-speech/UTMOS22) strong predictor to estimate MOS score of reconstructed audio
- For "Original wav" row: run UTMOS on the original waveforms

---

## Output Tables

Produce **two separate tables** (one per dataset), formatted as:

| Setup | LogMel diff. (ℓ₁) ↓ | WER ↓ | UTMOS ↑ |
|---|---|---|---|
| Original wav | - | ... | ... |
| Full RVQ levels | ... | ... | ... |
| 5 RVQ levels | ... | ... | ... |
| 3 RVQ levels | ... | ... | ... |
| 2 RVQ levels | ... | ... | ... |
| 1 RVQ level | ... | ... | ... |

---

## Project Structure
```
assignment04/
├── README.md           # Required: describe how to run
├── pyproject.toml      # uv-managed dependencies
├── eval.py             # Main evaluation script
├── metrics/
│   ├── logmel.py       # LogMel diff computation
│   ├── wer.py          # ASR + WER computation
│   └── utmos.py        # UTMOS scoring wrapper
└── results/
    ├── ljspeech_results.csv
    └── vctk_results.csv
```

---

## Key Implementation Notes

- **RVQ level control**: For EnCodec, pass `n_q=<N>` to `model.encode()` or truncate the returned codes to first N quantizers before decoding. For DAC, use `model.quantizer.n_codebooks` or pass codes sliced to `[:N]`.
- **Batching**: Process files in batches to avoid OOM; audio lengths vary.
- **Resampling**: Use `torchaudio.functional.resample` or `librosa.resample` to 24 kHz before any codec call.
- **LogMel**: Apply `torch.log` (or `log1p` for numerical stability) after `torchaudio.transforms.MelSpectrogram`. Normalize by total T×F elements.
- **Averaging**: Report metrics **averaged over all files** in each dataset/split.

---

## Recommended Dependencies (uv)
```
uv add torch torchaudio encodec dac-audio transformers jiwer utmos librosa soundfile numpy pandas
```
- `encodec` — Meta's EnCodec
- `dac-audio` — Descript Audio Codec
- `jiwer` — WER computation
- `transformers` — Whisper for ASR

---
