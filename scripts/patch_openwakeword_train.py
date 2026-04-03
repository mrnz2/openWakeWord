"""Patch openWakeWord train.py (v0.6.x) for Colab/Docker: skip broken TFLite step; clearer empty-clip error.

Idempotent: można uruchomić wielokrotnie na tym samym pliku.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")

    if "convert_onnx_to_tflite(os.path.join" in text:
        before = text.count("convert_onnx_to_tflite(os.path.join")
        pattern = re.compile(
            r"convert_onnx_to_tflite\(os\.path\.join\(config\[\"output_dir\"\], "
            r'config\["model_name"\] \+ "\.onnx"\),\s*'
            r'os\.path\.join\(config\["output_dir"\], config\["model_name"\] \+ "\.tflite"\)\)',
            re.MULTILINE,
        )
        repl = 'print("Skipping ONNX->TFLite in container; ONNX export complete.")'
        text, n = pattern.subn(repl, text, count=1)
        if n != 1 or before < 1:
            raise SystemExit(
                f"patch_openwakeword_train: expected 1 TFLite replacement, got {n} "
                f"(call sites before: {before})"
            )

    guard = "brak plików .wav w katalogu positive test"
    if guard not in text:
        needle = (
            "    positive_clips = [str(i) for i in Path(positive_test_output_dir).glob(\"*.wav\")]\n"
            "    duration_in_samples = []\n"
        )
        insert = (
            "    positive_clips = [str(i) for i in Path(positive_test_output_dir).glob(\"*.wav\")]\n"
            "    if not positive_clips:\n"
            "        raise RuntimeError(\n"
            '            "openWakeWord train: brak plików .wav w katalogu positive test — generacja TTS nie utworzyła próbek (espeak-ng, CUDA OOM, błąd Piper)."\n'
            "        )\n"
            "    duration_in_samples = []\n"
        )
        if needle not in text:
            raise SystemExit(
                "patch_openwakeword_train: brak oczekiwanego bloku positive_clips/duration_in_samples "
                "(inna wersja openWakeWord?)"
            )
        text = text.replace(needle, insert, 1)

    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
