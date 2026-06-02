"""
LogMel L1 difference metric.

D_mel = (1 / T*F) * || log MelSpec(x) - log MelSpec(x_hat) ||_1

Mel config: 24 kHz, n_fft=1024, win=1024, hop=256, n_mels=80, f_max=12000
"""
import torch
import torchaudio

_mel_fn: torchaudio.transforms.MelSpectrogram | None = None


def _get_mel_fn() -> torchaudio.transforms.MelSpectrogram:
    global _mel_fn
    if _mel_fn is None:
        _mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=80,
            f_min=0.0,
            f_max=12000.0,
            power=2.0,
        )
    return _mel_fn


def compute_logmel_diff(orig: torch.Tensor, recon: torch.Tensor) -> float:
    """
    orig, recon: (1, T) tensors at 24 kHz.
    Returns D_mel scalar (lower is better).
    """
    mel_fn = _get_mel_fn()

    def logmel(wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mel = mel_fn(wav)               # (1, 80, T_frames)
        return torch.log(mel.clamp(min=1e-5))

    lm_orig = logmel(orig.cpu())
    lm_recon = logmel(recon.cpu())

    T = min(lm_orig.shape[-1], lm_recon.shape[-1])
    return (lm_orig[..., :T] - lm_recon[..., :T]).abs().mean().item()
