#!/usr/bin/env python3
"""
Instalacja zależności pod Google Colab w bezpiecznej kolejności.

- Nie instaluje `openwakeword` z PyPI (klon v0.6.0 jest w colab_train.py).
- torch/torchaudio najpierw (CUDA 12.4 dla typowego GPU Colab, potem fallback PyPI).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, check: bool = True) -> int:
    print("\n$ " + " ".join(cmd) + "\n")
    return subprocess.run(cmd, check=check).returncode


def main() -> None:
    colab_dir = Path(__file__).resolve().parent
    train_req = colab_dir / "requirements-colab-train.txt"
    tflite_req = colab_dir / "requirements-colab-tflite.txt"
    py = sys.executable

    _run([py, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])

    # Colab GPU: koła cu124; przy CPU / błędzie — zwykły PyPI.
    torch_cmd = [
        py,
        "-m",
        "pip",
        "install",
        "torch",
        "torchaudio",
        "--index-url",
        "https://download.pytorch.org/whl/cu124",
    ]
    if _run(torch_cmd, check=False) != 0:
        print("\n[INFO] Instalacja torch z indeksu cu124 nie powiodła się — próbuję PyPI (CPU/CUDA z głównego indeksu).\n")
        _run([py, "-m", "pip", "install", "torch", "torchaudio"])

    _run([py, "-m", "pip", "install", "-r", str(train_req)])
    _run([py, "-m", "pip", "install", "-r", str(tflite_req)])
    _run([py, "-m", "pip", "install", "--force-reinstall", "numpy==1.26.4"])

    print("\nGotowe. Interpreter:", py)
    print("Uruchom: import torch; print(torch.__version__, torch.cuda.is_available())")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\n[install_colab_deps] Błąd (kod {exc.returncode}).", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
