"""WER scorer using Whisper-base for transcription."""
import torch
import torchaudio
import jiwer
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class WERScorer:
    def __init__(self, model_name: str = "openai/whisper-base", device: str = "cpu"):
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)

    def transcribe(self, wav: torch.Tensor, sr: int) -> str:
        """Transcribe waveform to text. wav: (1, T) at any sr."""
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        wav_np = wav.squeeze(0).float().cpu().numpy()

        inputs = self.processor(
            wav_np, sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                language="en",
                task="transcribe",
            )

        return self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

    @staticmethod
    def compute_wer(reference: str, hypothesis: str) -> float:
        ref = reference.strip().lower()
        hyp = hypothesis.strip().lower()
        if not ref:
            return 0.0
        if not hyp:
            return 1.0
        try:
            return jiwer.wer(ref, hyp)
        except Exception:
            return 1.0
