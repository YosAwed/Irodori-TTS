#!/usr/bin/env python3
"""
End-to-end CER + mel-MSE eval for Irodori-TTS WaveEx integration on the
duration-control checkpoint.

Uses kizuna-intelligence/Irodori-TTS-500M-v2-duration-control with phoneme-
based duration estimation (pyopenjtalk g2p × per-phone seconds) so each trial
has a fixed audio length and the trailing tail problem is gone. After a
warmup pass (so MIOpen kernel tuning doesn't pollute the first timed run)
runs an ODE-only baseline and a battery of WaveEx variants, reporting:

  * gen_seconds  — wall-clock for the sampling stage
  * speedup vs baseline
  * CER         — Whisper transcription vs reference text
  * mel_mse     — log-mel MSE vs the baseline audio (most discriminating)

Run with:
    PYTHONUNBUFFERED=1 HF_HOME=/mnt/hojo/hf_home \
    /home/yusuke/gitrepos/rocm-test/venv/bin/python -u eval_cer.py
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pyopenjtalk  # noqa: E402
import torch  # noqa: E402

from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    save_wav,
)
from irodori_tts.waveex import WaveExConfig  # noqa: E402


CHECKPOINT_PATH = (
    "/mnt/hojo/hf_home/hub/models--kizuna-intelligence--Irodori-TTS-500M-v2-duration-control/"
    "snapshots/f15b6bf01b3afa6dc4211680c4cfb16dcc102434/model.safetensors"
)
REF_WAV = "/home/yusuke/.claude/mera3.wav"
OUT_DIR = Path("/tmp/waveex_cer")
WHISPER_REPO = "openai/whisper-large-v3-turbo"

DEFAULT_TEXT = (
    "今日は朝からとても良い天気だったので、近くの公園まで散歩に出かけました。"
)
# Japanese TTS phone duration ~90 ms incl. pauses — gives a safety margin so
# the duration-control model has room for natural pacing without padding.
SECONDS_PER_PHONE = 0.10


@dataclass
class TrialConfig:
    name: str
    num_steps: int = 32
    waveex: WaveExConfig | None = None
    seed: int = 1234
    extra: dict[str, Any] = field(default_factory=dict)


def _strip_for_cer(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\s。、，．,.!?！？:：;；・「」『』（）()\"'`~^…\-]+", "", s)
    return s


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref = _strip_for_cer(reference)
    hyp = _strip_for_cer(hypothesis)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ref_i = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ref_i == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / max(1, len(ref))


def estimate_seconds_from_text(text: str) -> tuple[float, int]:
    phones = pyopenjtalk.g2p(text, kana=False).split()
    n = len(phones)
    return n * SECONDS_PER_PHONE, n


def log_mel(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    audio = audio.astype(np.float32)
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=16000, n_fft=1024, hop_length=256, n_mels=80, power=2.0
    )
    return np.log(np.maximum(mel, 1e-8)).astype(np.float32)


def mel_mse_vs_ref(
    audio: np.ndarray, sample_rate: int, ref_mel: np.ndarray
) -> float:
    mel = log_mel(audio, sample_rate)
    T = min(mel.shape[1], ref_mel.shape[1])
    return float(np.mean((mel[:, :T] - ref_mel[:, :T]) ** 2))


class WhisperASR:
    def __init__(self, device: str = "cuda") -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(WHISPER_REPO)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_REPO, dtype=self.dtype
        ).to(device)
        self.model.eval()

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        if sample_rate != 16000:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        device_inputs = {}
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                device_inputs[k] = v.to(self.device, dtype=self.dtype)
            else:
                device_inputs[k] = v.to(self.device)
        out = self.model.generate(
            **device_inputs,
            language="ja",
            task="transcribe",
            max_new_tokens=300,
            num_beams=1,
        )
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        return text.strip()


def build_trials() -> list[TrialConfig]:
    """
    Anchor + two production WaveEx schedules at 40 NFE.

    `paper6_spread_40` is the balanced winner: 6 full ODE refreshes spread
    across the CFG region (0,1,2,5,10,20), the rest direct-Taylor. ~3.9x DiT
    sample speedup, mel-MSE ~1.2, CER unchanged from baseline.
    """
    direct_kw = dict(wavelet="haar", taylor_order=1, history_size=2)

    trials: list[TrialConfig] = [
        TrialConfig(name="baseline_32_ode", num_steps=32, waveex=None),
        TrialConfig(
            name="paper_warm_40_taylor",
            num_steps=40,
            waveex=WaveExConfig(
                enabled=True, ode_step_indices=(0, 1, 2, 3, 5, 10, 20), **direct_kw
            ),
        ),
        TrialConfig(
            name="paper6_spread_40",
            num_steps=40,
            waveex=WaveExConfig(
                enabled=True, ode_step_indices=(0, 1, 2, 5, 10, 20), **direct_kw
            ),
        ),
    ]
    return trials


def run_one(
    runtime: InferenceRuntime,
    trial: TrialConfig,
    *,
    text: str,
    seconds: float,
) -> tuple[np.ndarray, int, float, dict[str, float]]:
    t0 = time.time()
    result = runtime.synthesize(
        SamplingRequest(
            text=text,
            ref_wav=REF_WAV,
            no_ref=False,
            num_candidates=1,
            num_steps=trial.num_steps,
            cfg_scale_text=3.0,
            cfg_scale_speaker=5.0,
            cfg_guidance_mode="independent",
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            seed=trial.seed,
            seconds=seconds,
            trim_tail=False,
            waveex=trial.waveex,
        ),
        log_fn=None,
    )
    gen_seconds = time.time() - t0

    audio = result.audio
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    stage = {name: float(sec) for name, sec in result.stage_timings}
    return audio, int(result.sample_rate), gen_seconds, stage


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[setup] device:", "cuda", torch.cuda.get_device_name(0))
    print("[setup] loading Irodori-TTS runtime (duration-control)...")
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=CHECKPOINT_PATH,
            model_device="cuda",
            codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
            model_precision="fp32",
            codec_device="cuda",
            codec_precision="fp32",
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            enable_watermark=False,
            compile_model=False,
            compile_dynamic=False,
        )
    )
    print("[setup] loading Whisper ASR...")
    asr = WhisperASR(device="cuda")

    text = DEFAULT_TEXT
    seconds, n_phones = estimate_seconds_from_text(text)
    print(f"[setup] reference text: {text}")
    print(f"[setup] phonemes: {n_phones}, predicted duration: {seconds:.2f}s "
          f"({SECONDS_PER_PHONE*1000:.0f}ms/phone)")

    trials = build_trials()

    # Warmup so MIOpen kernel tuning doesn't pollute the first timed run.
    # Two warmup passes: one ODE-only (to warm full forward pass) and one with
    # WaveEx (to warm anything that's only triggered when waveex is enabled).
    print("\n[warmup] 1/2 running 32-step ODE pass to warm MIOpen caches...")
    warmup_trial = TrialConfig(name="warmup", num_steps=32, waveex=None, seed=999)
    _, _, warmup_sec, _ = run_one(runtime, warmup_trial, text=text, seconds=seconds)
    print(f"[warmup] 1/2 done in {warmup_sec:.2f}s")
    print("[warmup] 2/2 running 16-step ODE pass (different num_steps to warm any size-conditional kernels)...")
    warmup_trial2 = TrialConfig(name="warmup2", num_steps=16, waveex=None, seed=998)
    _, _, warmup_sec2, _ = run_one(runtime, warmup_trial2, text=text, seconds=seconds)
    print(f"[warmup] 2/2 done in {warmup_sec2:.2f}s")

    summary = []
    baseline_mel: np.ndarray | None = None
    baseline_seconds: float | None = None
    baseline_sample_sec: float | None = None

    for trial in trials:
        out_wav = OUT_DIR / f"{trial.name}.wav"
        wx_indices = (
            None if trial.waveex is None or trial.waveex.ode_step_indices is None
            else len(trial.waveex.ode_step_indices)
        )
        print(
            f"\n=== {trial.name} (num_steps={trial.num_steps}, "
            f"waveex={trial.waveex is not None}, ode_count={wx_indices}) ==="
        )

        # Run twice and take the second (skip first to avoid any per-trial
        # warmup variance — kernels are warm but allocator state may differ).
        audio_np, sample_rate, gen_seconds, stage = run_one(
            runtime, trial, text=text, seconds=seconds
        )
        audio_np, sample_rate, gen_seconds, stage = run_one(
            runtime, trial, text=text, seconds=seconds
        )
        sample_sec = stage.get("sample_rf", float("nan"))
        save_wav(out_wav, torch.from_numpy(audio_np), sample_rate)
        stage_summary = ", ".join(f"{k}={v*1000:.0f}ms" for k, v in stage.items())
        print(f"  total={gen_seconds:.2f}s sample_rf={sample_sec:.2f}s -> {out_wav}")
        print(f"  stages: {stage_summary}")

        t0 = time.time()
        hyp = asr.transcribe(audio_np, sample_rate)
        asr_time = time.time() - t0
        cer = char_error_rate(text, hyp)

        if baseline_mel is None:
            baseline_mel = log_mel(audio_np, sample_rate)
            baseline_seconds = gen_seconds
            baseline_sample_sec = sample_sec
            mel_mse = 0.0
        else:
            mel_mse = mel_mse_vs_ref(audio_np, sample_rate, baseline_mel)

        speedup_total = (baseline_seconds / gen_seconds) if baseline_seconds else 1.0
        speedup_sample = (
            (baseline_sample_sec / sample_sec)
            if baseline_sample_sec and sample_sec > 0
            else 1.0
        )

        print(f"  asr ({asr_time:.2f}s): {hyp!r}")
        print(
            f"  CER = {cer*100:.2f}%   mel_mse = {mel_mse:.4f}   "
            f"speedup_total = {speedup_total:.2f}x   speedup_sample = {speedup_sample:.2f}x"
        )

        summary.append(
            {
                "name": trial.name,
                "num_steps": trial.num_steps,
                "ode_count_effective": (
                    trial.num_steps if trial.waveex is None
                    else (
                        wx_indices if wx_indices is not None
                        else trial.num_steps
                    )
                ),
                "gen_seconds": gen_seconds,
                "sample_seconds": sample_sec,
                "stage_seconds": stage,
                "speedup_total": speedup_total,
                "speedup_sample": speedup_sample,
                "asr_seconds": asr_time,
                "cer": cer,
                "hypothesis": hyp,
                "mel_mse": mel_mse,
                "wav": str(out_wav),
                "waveex": None if trial.waveex is None else {
                    "wavelet": trial.waveex.wavelet,
                    "taylor_order": trial.waveex.taylor_order,
                    "history_size": trial.waveex.history_size,
                    "ode_step_count": wx_indices,
                    "ode_step_indices": list(trial.waveex.ode_step_indices or ()),
                    "high_freq_mode": trial.waveex.high_freq_mode,
                },
            }
        )

    print("\n========== SUMMARY ==========")
    print(f"text: {text}")
    print(f"phones: {n_phones}, requested seconds: {seconds:.2f}")
    print(
        f"{'name':28s} {'total_s':>7s} {'samp_s':>7s} {'sp_tot':>7s} {'sp_smp':>7s} "
        f"{'cer%':>6s} {'mel':>6s}  hyp"
    )
    for row in summary:
        print(
            f"{row['name']:28s} {row['gen_seconds']:7.2f} {row['sample_seconds']:7.2f} "
            f"{row['speedup_total']:6.2f}x {row['speedup_sample']:6.2f}x "
            f"{row['cer']*100:6.2f} {row['mel_mse']:6.3f}  {row['hypothesis'][:40]}"
        )

    out_json = OUT_DIR / "summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[wrote] {out_json}")


if __name__ == "__main__":
    main()
