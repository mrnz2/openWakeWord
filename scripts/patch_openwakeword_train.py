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

    # Adversarial negatives use batch_size=tts_batch_size//7; values < 7 yield 0 and no WAVs -> StopIteration in features.
    old_neg_bs = 'batch_size=config["tts_batch_size"]//7,'
    new_neg_bs = 'batch_size=max(1, config["tts_batch_size"]//7),'
    if old_neg_bs in text:
        text = text.replace(old_neg_bs, new_neg_bs)

    # n_total must match augmented clip list (len * augmentation_rounds), not raw listdir count.
    n_total_repls = (
        (
            "compute_features_from_generator(positive_clips_train_generator, n_total=len(os.listdir(positive_train_output_dir)),",
            "compute_features_from_generator(positive_clips_train_generator, n_total=len(positive_clips_train),",
        ),
        (
            "compute_features_from_generator(negative_clips_train_generator, n_total=len(os.listdir(negative_train_output_dir)),",
            "compute_features_from_generator(negative_clips_train_generator, n_total=len(negative_clips_train),",
        ),
        (
            "compute_features_from_generator(positive_clips_test_generator, n_total=len(os.listdir(positive_test_output_dir)),",
            "compute_features_from_generator(positive_clips_test_generator, n_total=len(positive_clips_test),",
        ),
        (
            "compute_features_from_generator(negative_clips_test_generator, n_total=len(os.listdir(negative_test_output_dir)),",
            "compute_features_from_generator(negative_clips_test_generator, n_total=len(negative_clips_test),",
        ),
    )
    for old, new in n_total_repls:
        if old in text:
            text = text.replace(old, new, 1)

    neg_train_guard = "brak plików .wav w katalogu negative_train"
    if neg_train_guard not in text:
        needle_nt = (
            '            negative_clips_train = [str(i) for i in Path(negative_train_output_dir).glob("*.wav")]'
            '*config["augmentation_rounds"]\n'
            '            negative_clips_train_generator = augment_clips(negative_clips_train, total_length=config["total_length"],\n'
        )
        insert_nt = (
            '            negative_clips_train = [str(i) for i in Path(negative_train_output_dir).glob("*.wav")]'
            '*config["augmentation_rounds"]\n'
            "            if not negative_clips_train:\n"
            "                raise RuntimeError(\n"
            '                    "openWakeWord train: brak plików .wav w katalogu negative_train — generacja adversarial negative '
            'nie utworzyła próbek (np. tts_batch_size//7 == 0; ustaw tts_batch_size >= 7 lub zastosuj patch max(1, ...))."\n'
            "                )\n"
            '            negative_clips_train_generator = augment_clips(negative_clips_train, total_length=config["total_length"],\n'
        )
        if needle_nt not in text:
            raise SystemExit(
                "patch_openwakeword_train: brak oczekiwanego bloku negative_clips_train/augment_clips "
                "(inna wersja openWakeWord?)"
            )
        text = text.replace(needle_nt, insert_nt, 1)

    neg_test_guard = "brak plików .wav w katalogu negative_test"
    if neg_test_guard not in text:
        needle_nx = (
            '            negative_clips_test = [str(i) for i in Path(negative_test_output_dir).glob("*.wav")]'
            '*config["augmentation_rounds"]\n'
            '            negative_clips_test_generator = augment_clips(negative_clips_test, total_length=config["total_length"],\n'
        )
        insert_nx = (
            '            negative_clips_test = [str(i) for i in Path(negative_test_output_dir).glob("*.wav")]'
            '*config["augmentation_rounds"]\n'
            "            if not negative_clips_test:\n"
            "                raise RuntimeError(\n"
            '                    "openWakeWord train: brak plików .wav w katalogu negative_test — generacja adversarial negative '
            'nie utworzyła próbek (np. tts_batch_size//7 == 0; ustaw tts_batch_size >= 7)."\n'
            "                )\n"
            '            negative_clips_test_generator = augment_clips(negative_clips_test, total_length=config["total_length"],\n'
        )
        if needle_nx not in text:
            raise SystemExit(
                "patch_openwakeword_train: brak oczekiwanego bloku negative_clips_test/augment_clips "
                "(inna wersja openWakeWord?)"
            )
        text = text.replace(needle_nx, insert_nx, 1)

    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
