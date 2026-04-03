#!/usr/bin/env python3
"""
Instalacja zależności pod Google Colab w bezpiecznej kolejności.

- Nie instaluje `openwakeword` z PyPI (klon v0.6.0 jest w colab_train.py).
- Każda linia z requirements-colab-train.txt = osobne `pip install` — przy błędzie
  od razu widać winnego pakietu (np. webrtcvad, piper-phonemize).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _pip(py: str, args: list[str], *, label: str) -> None:
    cmd = [py, "-m", "pip", "install", *args]
    print("\n" + "=" * 72)
    print(f"ETAP: {label}")
    print("$ " + " ".join(cmd))
    print("=" * 72 + "\n")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(
            f"\n*** BŁĄD: etap „{label}” zakończył się kodem {r.returncode}. ***\n"
            "Jeśli to kompilacja (webrtcvad): w Colab uruchom wcześniej komórkę apt z\n"
            "  build-essential python3-dev\n",
            file=sys.stderr,
        )
        raise SystemExit(r.returncode)


def _install_requirements_lines(py: str, req_path: Path, *, label_prefix: str) -> None:
    text = req_path.read_text(encoding="utf-8")
    n = 0
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        n += 1
        _pip(py, [line], label=f"{label_prefix} [{n}] {line[:60]}…" if len(line) > 60 else f"{label_prefix} [{n}] {line}")


def main() -> None:
    colab_dir = Path(__file__).resolve().parent
    train_req = colab_dir / "requirements-colab-train.txt"
    tflite_req = colab_dir / "requirements-colab-tflite.txt"
    py = sys.executable

    if not train_req.is_file():
        print(f"Brak pliku: {train_req}", file=sys.stderr)
        raise SystemExit(2)
    if not tflite_req.is_file():
        print(f"Brak pliku: {tflite_req}", file=sys.stderr)
        raise SystemExit(2)

    _pip(py, ["-U", "pip", "setuptools", "wheel"], label="Narzędzia pip")

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
    print("\n" + "=" * 72)
    print("ETAP: PyTorch (CUDA 12.4 — typowy GPU Colab)")
    print("$ " + " ".join(torch_cmd))
    print("=" * 72 + "\n")
    r = subprocess.run(torch_cmd)
    if r.returncode != 0:
        print("\n[INFO] cu124 nie zadziałał — instaluję torch+torchaudio z PyPI.\n")
        _pip(py, ["torch", "torchaudio"], label="PyTorch (fallback PyPI)")

    _install_requirements_lines(py, train_req, label_prefix="Trening")

    _install_requirements_lines(py, tflite_req, label_prefix="TFLite")

    _pip(py, ["--force-reinstall", "numpy==1.26.4"], label="Ponownie numpy==1.26.4 (po TF)")

    print("\nGotowe. Interpreter:", py)
    print("Test: import torch; print(torch.__version__, torch.cuda.is_available())")


if __name__ == "__main__":
    main()
