"""
Pomocnik smoke-testu: generuje 10 plików .wav z białym szumem w ./test_data_dummy.

Użycie w Colab:
    %run colab/generate_test_data_dummy.py
lub
    from colab.generate_test_data_dummy import generate_dummy_wavs
    generate_dummy_wavs()
"""
from __future__ import annotations

import struct
import wave
from pathlib import Path


def generate_dummy_wavs(
    out_dir: str | Path = "./test_data_dummy",
    n_files: int = 10,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
) -> list[Path]:
    """
    Tworzy `n_files` plików .wav z białym szumem (int16, mono, 16 kHz).

    Args:
        out_dir:     Folder docelowy (zostanie utworzony jeśli nie istnieje).
        n_files:     Liczba plików do wygenerowania.
        duration_s:  Czas trwania każdego pliku w sekundach.
        sample_rate: Częstotliwość próbkowania (domyślnie 16 000 Hz).

    Returns:
        Lista ścieżek do wygenerowanych plików.
    """
    import random

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    n_samples = int(sample_rate * duration_s)
    created: list[Path] = []

    for i in range(n_files):
        path = target / f"dummy_{i:02d}.wav"
        raw_samples = [random.randint(-1000, 1000) for _ in range(n_samples)]
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16 = 2 bajty
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{n_samples}h", *raw_samples))
        created.append(path)
        print(f"  Zapisano: {path}")

    print(f"\nGotowe: {len(created)} plików w {target.resolve()}")
    return created


if __name__ == "__main__":
    generate_dummy_wavs()
