"""UTMOS22 MOS predictor wrapper."""
import copy
import dataclasses as _dc
import sys
import types
import torch

# ── Python 3.11 + fairseq/hydra compatibility ─────────────────────────────────
# fairseq 0.12.2 uses mutable dataclass instances as field defaults, which
# Python 3.11 now rejects.  We patch _process_class to convert them to
# default_factory BEFORE _get_field ever sees them — covering both:
#   encoder: EncDecBaseConfig = EncDecBaseConfig()          (direct assignment)
#   quant_noise: QuantNoiseConfig = field(default=QNC())   (field() wrapper)
_orig_process_class = _dc._process_class


def _compat_process_class(cls, *args, **kwargs):
    own = cls.__dict__.get("__annotations__", {})
    for name in list(own):
        val = cls.__dict__.get(name, _dc.MISSING)
        if val is _dc.MISSING:
            continue

        if isinstance(val, _dc.Field):
            # Pattern 2: field(default=<dataclass instance>)
            if (val.default is not _dc.MISSING
                    and not isinstance(val.default, _dc.MISSING.__class__)
                    and hasattr(val.default, "__dataclass_fields__")):
                captured = val.default
                val.default = _dc.MISSING
                val.default_factory = lambda v=captured: copy.copy(v)  # type: ignore[assignment]
        elif (not isinstance(val, (_dc.Field, type(None), int, float, str, bool, bytes, tuple, frozenset))
              and hasattr(val, "__dataclass_fields__")):
            # Pattern 1: direct mutable default
            captured = val
            try:
                setattr(cls, name, _dc.field(default_factory=lambda v=captured: copy.copy(v)))
            except Exception:
                pass

    return _orig_process_class(cls, *args, **kwargs)


_dc._process_class = _compat_process_class

# fairseq calls hydra_init() at module level; omegaconf rejects the patched
# MISSING values there. hydra_init is only needed for the CLI, so stub it out.
if "fairseq" not in sys.modules:
    _fake_init = types.ModuleType("fairseq.dataclass.initialize")
    _fake_init.hydra_init = lambda: None
    sys.modules.setdefault("fairseq.dataclass.initialize", _fake_init)


class UTMOSScorer:
    def __init__(self):
        import utmos
        self.model = utmos.Score()

    def score(self, wav: torch.Tensor, sr: int) -> float:
        """Predict MOS score (1–5 scale). wav: (C, T) or (T,) tensor."""
        wav_t = wav.float().cpu()
        if wav_t.dim() == 1:
            wav_t = wav_t.unsqueeze(0)          # → (1, T)
        elif wav_t.shape[0] > 1:
            wav_t = wav_t.mean(dim=0, keepdim=True)  # → (1, T)
        return float(self.model.calculate_wav(wav_t, sr))
