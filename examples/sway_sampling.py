"""Minimum end-to-end example: synthesize Japanese speech with Irodori-TTS-v2
using Sway Sampling (training-free inference-time schedule, F5-TTS style).

Usage:
    python examples/sway_sampling.py path/to/reference.wav "合成したいテキスト"

Recommended parameters (num_steps=6, t_schedule_mode="sway", sway_coeff=-1.0)
are chosen by sweeping CER / speaker similarity over candidate settings and
picking the best operating point. Details:
    https://magazine.kizuna-intelligence.com/articles/article-d9ac7ce68a98
"""
from __future__ import annotations

import sys

import soundfile as sf

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    ref_wav, text = sys.argv[1], sys.argv[2]

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint="Aratako/Irodori-TTS-500M-v2",
            model_device="cuda",
            model_precision="bf16",
            codec_device="cuda",
            codec_precision="bf16",
        )
    )

    out = runtime.synthesize(
        SamplingRequest(
            text=text,
            ref_wav=ref_wav,
            seconds=8.0,
            num_steps=6,
            t_schedule_mode="sway",
            sway_coeff=-1.0,
            seed=42,
        ),
    )

    audio = out.audio[0].squeeze().float().cpu().numpy()
    sf.write("output.wav", audio, runtime.codec.sample_rate)
    print(f"saved: output.wav ({len(audio) / runtime.codec.sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
