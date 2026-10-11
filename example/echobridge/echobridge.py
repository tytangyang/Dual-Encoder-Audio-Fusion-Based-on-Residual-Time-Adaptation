# example/echobridge4/echobridgeup4.py
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperProcessor

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from models.audiotransformer import ced_tiny, ced_mini, ced_small, ced_base
from models.layers import to_2tuple


# -----------------------------
# Helpers
# -----------------------------
def length_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    if max_len is None:
        max_len = int(lengths.amax().item())
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return (idx < lengths.unsqueeze(1)).long()


def _mel_frames_from_audio_len(audio_len: torch.Tensor, *, n_fft: int, hop_length: int, center: bool) -> torch.Tensor:
    pad = (n_fft // 2) if center else 0
    numer = torch.clamp(audio_len + 2 * pad - n_fft, min=0)
    return 1 + (numer // hop_length)


def _conv_out_len(L: torch.Tensor, kernel: int, stride: int) -> torch.Tensor:
    out = (L - kernel) // stride + 1
    return torch.clamp(out, min=0)


def _align_to_T(feat: torch.Tensor, mask: torch.Tensor, T: int) -> tuple[torch.Tensor, torch.Tensor]:
    B, T0, D = feat.shape
    if T0 == T:
        return feat, mask[:, :T]
    if T0 > T:
        return feat[:, :T], mask[:, :T]

    x = feat.transpose(1, 2)  # (B, D, T0)
    x = F.interpolate(x, size=T, mode="linear", align_corners=False)
    feat2 = x.transpose(1, 2)

    m = mask.float().unsqueeze(1)  # (B, 1, T0)
    m = F.interpolate(m, size=T, mode="nearest")
    mask2 = m.squeeze(1).long()
    return feat2, mask2


# -----------------------------
# Whisper sub-encoder (NOT ending with Encoder)
# -----------------------------
class WhisperEnc(nn.Module):
    def __init__(self, model_name: str = "openai/whisper-base"):
        super().__init__()
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name).get_encoder()
        self.output_dim = int(self.model.config.d_model)

        self.sampling_rate = int(self.processor.feature_extractor.sampling_rate)
        hop = int(self.processor.feature_extractor.hop_length)
        # Whisper encoder time base (two conv strides => factor 4)
        self.hop_size_in_ms = (hop * 4 / self.sampling_rate) * 1000.0  # ~40ms at 16k

    def forward(self, audio: torch.Tensor, audio_attention_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(audio, torch.Tensor)
        audio_np = audio.detach().cpu().numpy()

        if audio_np.ndim == 1:
            audio_list = [audio_np]
        elif audio_np.ndim == 2:
            audio_list = [a for a in audio_np]
        else:
            raise ValueError("Audio tensor must be 1D (single) or 2D (batch).")

        if audio_attention_mask is None:
            audio_lens = torch.tensor([a.shape[-1] for a in audio_list], dtype=torch.long)
        else:
            audio_lens = audio_attention_mask.sum(-1).long().cpu()

        mel_lengths = audio_lens // int(self.processor.feature_extractor.hop_length)
        feature_lengths = (mel_lengths - 1) // 2 + 1
        feature_lengths = (feature_lengths - 1) // 2 + 1
        trim_length = int(feature_lengths.amax().item())
        attention_mask = length_to_mask(feature_lengths, max_len=trim_length)

        features = self.processor(audio_list, sampling_rate=self.sampling_rate, return_tensors="pt").to(self.model.device)
        output = self.model(**features).last_hidden_state
        return output[:, :trim_length, :], attention_mask.to(output.device)


# -----------------------------
# CED sub-encoder (NOT ending with Encoder)
# -----------------------------
class CEDEEnc(nn.Module):
    def __init__(self, model_name: str = "ced_mini", pretrained: bool = True):
        super().__init__()
        entry = {
            "ced_tiny": ced_tiny,
            "ced_mini": ced_mini,
            "ced_small": ced_small,
            "ced_base": ced_base,
        }[model_name]

        self.model = entry(pretrained=pretrained)
        self.model.eval()
        self.output_dim = int(self.model.embed_dim)

        self.hop_length = int(self.model.hop_size)
        self.n_fft = 512
        self.center = bool(getattr(self.model, "center", True))
        self.target_mel_len = int(self.model.target_length)

        ps = to_2tuple(self.model.patch_size)
        st = to_2tuple(self.model.patch_stride)
        self.patch_size_t = int(ps[-1])
        self.patch_stride_t = int(st[-1])
        self.f_grid = int(self.model.patch_embed.grid_size[0])

    @torch.no_grad()
    def forward(self, audio: torch.Tensor, audio_attention_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(audio, torch.Tensor)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.ndim != 2:
            raise ValueError("Audio tensor must be 1D or 2D.")

        B, Lmax = audio.shape
        dev = next(self.model.parameters()).device
        audio = audio.to(dev)

        if audio_attention_mask is None:
            audio_lens = torch.full((B,), Lmax, device=dev, dtype=torch.long)
        else:
            audio_attention_mask = audio_attention_mask.to(dev)
            audio_lens = audio_attention_mask.sum(-1).long()

        mel_lens = _mel_frames_from_audio_len(audio_lens, n_fft=self.n_fft, hop_length=self.hop_length, center=self.center)
        mel_lens = torch.clamp(mel_lens, max=self.target_mel_len)
        mel_trim = int(mel_lens.amax().item())

        spec = self.model.front_end(audio)  # (B, n_mels, T_full)
        if spec.shape[-1] < mel_trim:
            pad = spec.new_zeros(B, spec.shape[1], mel_trim)
            pad[..., : spec.shape[-1]] = spec
            spec = pad
        else:
            spec = spec[..., :mel_trim]

        x = spec.unsqueeze(1)
        x = self.model.init_bn(x)
        tokens = self.model.forward_features(x)  # (B, N, D)

        N = tokens.shape[1]
        t_grid = N // self.f_grid
        feat_2d = tokens.view(B, self.f_grid, t_grid, self.output_dim)
        feat = feat_2d.mean(dim=1)  # (B, t, D)

        t_lens = _conv_out_len(mel_lens, kernel=self.patch_size_t, stride=self.patch_stride_t)
        t_lens = torch.clamp(t_lens, max=feat.shape[1])
        trim_t = int(t_lens.amax().item())
        attn_mask = length_to_mask(t_lens, max_len=trim_t).to(feat.device)

        return feat[:, :trim_t, :], attn_mask


# -----------------------------
# Multi-scale temporal adapter (zero-init)
# -----------------------------
class _DWConv1d(nn.Module):
    def __init__(self, d: int, k: int):
        super().__init__()
        assert k % 2 == 1
        self.dw = nn.Conv1d(d, d, kernel_size=k, padding=k // 2, groups=d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw(x)


class _SEGate(nn.Module):
    """
    Squeeze-Excitation over channels, but zero-init output so initial gate=0 effect.
    """
    def __init__(self, d: int, r: int = 8):
        super().__init__()
        h = max(4, d // r)
        self.fc1 = nn.Linear(d, h, bias=True)
        self.fc2 = nn.Linear(h, d, bias=True)
        # zero-init final so output == 0 at init
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        s = x.mean(dim=1)            # (B,D)
        s = F.gelu(self.fc1(s))
        s = self.fc2(s)              # (B,D) -> 0 at init
        return torch.tanh(s).unsqueeze(1)  # (B,1,D), still 0 at init


class _MultiScaleTemporalAdapter(nn.Module):
    """
    delta = PW( DWk3(LN(x)) + DWk7(LN(x)) )
    then channel gate (SE) to modulate delta
    All zero-init at output => delta==0 at init
    """
    def __init__(self, d: int, dropout: float = 0.0, k1: int = 3, k2: int = 7):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

        self.dw1 = _DWConv1d(d, k1)
        self.dw2 = _DWConv1d(d, k2)
        self.pw = nn.Conv1d(d, d, kernel_size=1, bias=False)
        nn.init.zeros_(self.pw.weight)  # ensure output 0 at init

        self.se = _SEGate(d, r=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = self.drop(y)
        y = y.transpose(1, 2)                # (B,D,T)
        y = self.dw1(y) + self.dw2(y)        # (B,D,T)
        y = self.pw(y)                        # (B,D,T) -> 0 at init
        y = y.transpose(1, 2)                # (B,T,D)
        g = self.se(x)                        # (B,1,D) -> 0 at init
        return y * (1.0 + g)                  # still 0 at init


# ---------------------------------------------------------
# Up2 Encoder (ONLY class ending with Encoder)
# ---------------------------------------------------------
class EchobridgeUp2Encoder(nn.Module):
    """
    Strict baseline: Concat+Linear
    Upgrade: Multi-scale temporal conv residual adapter + channel-wise alpha (all-zero init)

    API:
      forward(audio, audio_attention_mask=None) -> (embeddings[B,T,D], attention_mask[B,T])
    """

    def __init__(
        self,
        whisper_name: str = "openai/whisper-base",
        ced_name: str = "ced_mini",
        d_llm: int = 1024,
        pretrained_ced: bool = True,
        dropout: float = 0.1,
        adapter_dropout: float = 0.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.whisper = WhisperEnc(whisper_name)
        self.ced = CEDEEnc(ced_name, pretrained=pretrained_ced)

        # Baseline projection
        self.w2h = nn.Linear(self.whisper.output_dim, d_llm)
        self.c2h = nn.Linear(self.ced.output_dim, d_llm)

        # Baseline fusion: Concat+Linear
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_llm * 2),
            nn.Dropout(dropout),
            nn.Linear(d_llm * 2, d_llm),
        )
        self.out_norm = nn.LayerNorm(d_llm)

        # Upgrade
        self.adapter = _MultiScaleTemporalAdapter(d_llm, dropout=adapter_dropout, k1=3, k2=7)
        self.alpha = nn.Parameter(torch.zeros(d_llm))  # all-zero init => strict baseline

        # Required public attrs (use Whisper time base)
        self.output_dim = int(d_llm)
        self.sampling_rate = int(self.whisper.sampling_rate)
        self.hop_size_in_ms = float(self.whisper.hop_size_in_ms)

        if freeze_base:
            for p in self.w2h.parameters():
                p.requires_grad = False
            for p in self.c2h.parameters():
                p.requires_grad = False
            for p in self.fuse.parameters():
                p.requires_grad = False
            for p in self.out_norm.parameters():
                p.requires_grad = False

    def forward(self, audio: torch.Tensor, audio_attention_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        w_feat, w_mask = self.whisper(audio, audio_attention_mask)
        c_feat, c_mask = self.ced(audio, audio_attention_mask)

        T = w_feat.shape[1]
        c_feat, c_mask = _align_to_T(c_feat, c_mask, T)
        w_mask = w_mask[:, :T].to(w_feat.device)
        fused_mask = (w_mask & c_mask).long()

        # baseline
        w_h = self.w2h(w_feat)
        c_h = self.c2h(c_feat)
        base = self.fuse(torch.cat([w_h, c_h], dim=-1))
        base = self.out_norm(base)

        # upgrade (delta==0 at init)
        delta = self.adapter(base)
        out = base + delta * self.alpha.view(1, 1, -1)

        return out, fused_mask
