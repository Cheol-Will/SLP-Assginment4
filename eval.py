"""
RVQ-based NAC evaluation.

Encodes and decodes audio at varying RVQ levels using DAC 24 kHz,
then reports LogMel L1 diff, WER, and UTMOS for each setting.

Usage:
    uv run eval.py
    uv run eval.py --vctk-dir /path/to/VCTK-Corpus-0.92 --limit 10
"""
import argparse
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

import dac
from metrics.logmel import compute_logmel_diff
from metrics.wer import WERScorer
from metrics.utmos import UTMOSScorer

# ── Paths ─────────────────────────────────────────────────────────────────────
LJ_MANIFEST = "/home/cheolseok/audio/SLP-Assignment3/manifests/ljspeech_ipa_compressed.jsonl"
LJ_BASE_DIR = "/home/cheolseok/audio/SLP-Assignment3"

# ── RVQ configurations ────────────────────────────────────────────────────────
# None → use all available codebooks (full)
RVQ_CONFIGS = [
    ("Full RVQ levels", None),
    ("5 RVQ levels",    5),
    ("3 RVQ levels",    3),
    ("2 RVQ levels",    2),
    ("1 RVQ level",     1),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Audio helpers ──────────────────────────────────────────────────────────────

def load_audio_mono(path: str, target_sr: int = 24000) -> torch.Tensor:
    """Load any audio file, downmix to mono, resample to target_sr.
    Returns (1, T) float32 tensor."""
    import soundfile as sf
    import numpy as np

    data, sr = sf.read(path, dtype="float32", always_2d=True)   # (T, C)
    wav = torch.from_numpy(data.T)                               # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav  # (1, T)


def encode_decode(model: dac.DAC, wav_24k: torch.Tensor, n_q: int | None) -> torch.Tensor:
    """
    Round-trip through DAC with n_q quantizers.
    wav_24k : (1, T) at 24 kHz
    Returns  : (1, T) at 24 kHz, trimmed to the original length
    """
    orig_len = wav_24k.shape[-1]
    x = wav_24k.unsqueeze(0).to(DEVICE)          # (1, 1, T)
    x = model.preprocess(x, 24000)               # pad to hop multiple → (1, 1, T_pad)

    with torch.no_grad():
        z, codes, latents, _, _ = model.encode(x, n_quantizers=n_q)
        y = model.decode(z)                      # (1, 1, T_pad)

    return y.squeeze(0).cpu()[..., :orig_len]    # (1, T)


# ── Dataset loaders ────────────────────────────────────────────────────────────

def get_ljspeech_val() -> list[tuple[str, str]]:
    """100-file LJSpeech validation split (indices 12500–12599)."""
    with open(LJ_MANIFEST) as f:
        records = [json.loads(line) for line in f if line.strip()]

    val = records[12500:12600]
    return [
        (
            os.path.join(LJ_BASE_DIR, r["wav_path"]),
            r.get("normalized_transcript", r.get("transcript", "")),
        )
        for r in val
        if r.get("wav_present", True)
    ]


def get_vctk_data(data_dir: str, speakers: tuple = ("p232", "p257")) -> list[tuple[str, str]]:
    """
    Return (wav_path, transcript) pairs for the given VCTK speakers.

    Expects the VCTK-Corpus-0.92 layout:
        <data_dir>/wav48_silence_trimmed/<speaker>/<speaker>_NNN_mic2.flac
        <data_dir>/txt/<speaker>/<speaker>_NNN.txt

    If the directory does not exist, downloads via torchaudio.datasets.VCTK_092.
    """
    root = Path(data_dir)

    # Support both: bare root that IS the corpus, or root that CONTAINS VCTK-Corpus-0.92/
    if (root / "wav48_silence_trimmed").exists():
        vctk_root = root
    elif (root / "VCTK-Corpus-0.92" / "wav48_silence_trimmed").exists():
        vctk_root = root / "VCTK-Corpus-0.92"
    else:
        print(f"VCTK not found at {data_dir}. Downloading via torchaudio (~11 GB)…")
        torchaudio.datasets.VCTK_092(root=str(root), download=True)
        # After download torchaudio places data at root/VCTK-Corpus-0.92
        vctk_root = root / "VCTK-Corpus-0.92"

    wav_root = vctk_root / "wav48_silence_trimmed"
    txt_root = vctk_root / "txt"

    files: list[tuple[str, str]] = []
    for speaker in speakers:
        wav_dir = wav_root / speaker
        txt_dir = txt_root / speaker

        if not wav_dir.exists():
            print(f"Warning: no audio directory for {speaker} at {wav_dir}")
            continue

        # Prefer mic2; fall back to mic1, then any wav/flac
        wav_files = sorted(wav_dir.glob(f"{speaker}_*_mic2.flac"))
        if not wav_files:
            wav_files = sorted(wav_dir.glob(f"{speaker}_*_mic1.flac"))
        if not wav_files:
            wav_files = sorted(wav_dir.glob(f"{speaker}_*.flac"))
        if not wav_files:
            wav_files = sorted(wav_dir.glob(f"{speaker}_*.wav"))

        for wav_file in wav_files:
            # p232_001_mic2 → p232_001
            stem_parts = wav_file.stem.split("_")
            utt_id = f"{stem_parts[0]}_{stem_parts[1]}"
            txt_file = txt_dir / f"{utt_id}.txt"
            transcript = txt_file.read_text().strip() if txt_file.exists() else ""
            files.append((str(wav_file), transcript))

    return files



def evaluate_dataset(
    codec: dac.DAC,
    data: list[tuple[str, str]],
    wer_scorer: WERScorer,
    utmos_scorer: UTMOSScorer,
    dataset_name: str,
    limit: int | None = None,
) -> pd.DataFrame:

    if limit:
        data = data[:limit]

    # Accumulators: {setup_name: {metric: [values]}}
    codec_acc: dict[str, dict[str, list[float]]] = {
        name: {"logmel": [], "wer": [], "utmos": []}
        for name, _ in RVQ_CONFIGS
    }
    orig_wer_list:   list[float] = []
    orig_utmos_list: list[float] = []

    for wav_path, transcript in tqdm(data, desc=dataset_name, dynamic_ncols=True):
        try:
            wav = load_audio_mono(wav_path)      # (1, T) at 24 kHz
        except Exception as e:
            print(f"  [skip] load failed: {wav_path}: {e}")
            continue

        asr_orig = wer_scorer.transcribe(wav, 24000)
        orig_wer_list.append(WERScorer.compute_wer(transcript, asr_orig))
        orig_utmos_list.append(utmos_scorer.score(wav, 24000))

        for setup_name, n_q in RVQ_CONFIGS:
            try:
                wav_hat = encode_decode(codec, wav, n_q)

                codec_acc[setup_name]["logmel"].append(compute_logmel_diff(wav, wav_hat))

                asr_hat = wer_scorer.transcribe(wav_hat, 24000)
                codec_acc[setup_name]["wer"].append(WERScorer.compute_wer(asr_orig, asr_hat))

                codec_acc[setup_name]["utmos"].append(utmos_scorer.score(wav_hat, 24000))

            except Exception as e:
                print(f"  [skip] {setup_name} failed on {Path(wav_path).name}: {e}")

    # ── Build result rows ────────────────────────────────────────────────────
    rows = []
    rows.append({
        "Setup":               "Original wav",
        "LogMel diff. (ℓ₁) ↓": "-",
        "WER ↓":               f"{np.mean(orig_wer_list):.4f}"   if orig_wer_list   else "N/A",
        "UTMOS ↑":             f"{np.mean(orig_utmos_list):.4f}" if orig_utmos_list else "N/A",
    })

    for setup_name, _ in RVQ_CONFIGS:
        r = codec_acc[setup_name]
        rows.append({
            "Setup":               setup_name,
            "LogMel diff. (ℓ₁) ↓": f"{np.mean(r['logmel']):.4f}" if r["logmel"] else "N/A",
            "WER ↓":               f"{np.mean(r['wer']):.4f}"     if r["wer"]    else "N/A",
            "UTMOS ↑":             f"{np.mean(r['utmos']):.4f}"   if r["utmos"]  else "N/A",
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="RVQ-level NAC analysis")
    parser.add_argument(
        "--vctk-dir",
        default="data/VCTK-Corpus-0.92",
        help="Path to VCTK-Corpus-0.92 (downloaded here if absent)",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only this many files per dataset (for quick testing)",
    )
    args = parser.parse_args()

    global DEVICE
    if args.device:
        DEVICE = args.device
    print(f"Device: {DEVICE}")

    print("Loading DAC 24 kHz model…")
    model_path = dac.utils.download(model_type="24khz")
    codec = dac.DAC.load(model_path)
    codec.eval()
    codec.to(DEVICE)
    n_cb = codec.quantizer.n_codebooks
    print(f"  → {n_cb} codebooks available")

    print("Loading Whisper-base for WER…")
    wer_scorer = WERScorer(device=DEVICE)

    print("Loading UTMOS scorer…")
    utmos_scorer = UTMOSScorer()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== LJSpeech — Validation Split ===")
    lj_data = get_ljspeech_val()
    print(f"  {len(lj_data)} files")
    lj_df = evaluate_dataset(codec, lj_data, wer_scorer, utmos_scorer, "LJSpeech", args.limit)

    lj_csv = os.path.join(args.output_dir, "ljspeech_results.csv")
    lj_df.to_csv(lj_csv, index=False)
    print("\nLJSpeech Results:")
    print(lj_df.to_markdown(index=False))

    print("\n=== VCTK — Speakers p232 & p257 ===")
    vctk_data = get_vctk_data(args.vctk_dir)
    print(f"  {len(vctk_data)} files")
    vctk_df = evaluate_dataset(codec, vctk_data, wer_scorer, utmos_scorer, "VCTK", args.limit)

    vctk_csv = os.path.join(args.output_dir, "vctk_results.csv")
    vctk_df.to_csv(vctk_csv, index=False)
    print("\nVCTK Results:")
    print(vctk_df.to_markdown(index=False))

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
