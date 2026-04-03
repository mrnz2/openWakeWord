#!/usr/bin/env python3
"""
Instalacja zależności pod Google Colab w bezpiecznej kolejności.

- Nie instaluje `openwakeword` z PyPI (klon v0.6.0 jest w colab_train.py).
- Każda linia z requirements = osobne `pip install` — widać winnego pakietu.
- Przy błędzie: RuntimeError z ETAP + końcówka stderr/stdout pip (w tracebacku na dole komórki).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _pip(py: str, args: list[str], *, label: str) -> None:
    cmd = [py, "-m", "pip", "install", *args]
    print("\n" + "=" * 72, flush=True)
    print(f"ETAP: {label}", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    print("=" * 72 + "\n", flush=True)

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout, end="", flush=True)
    if r.stderr:
        print(r.stderr, end="", file=sys.stderr, flush=True)
    if r.returncode != 0:
        tail_e = (r.stderr or "")[-8000:]
        tail_o = (r.stdout or "")[-4000:]
        raise RuntimeError(
            f"Pip zwrócił kod {r.returncode}.\n"
            f"ETAP: {label}\n"
            f"Komenda: pip install {' '.join(args)}\n\n"
            f"--- stderr (koniec, max ~8k znaków) ---\n{tail_e}\n\n"
            f"--- stdout (koniec, max ~4k znaków) ---\n{tail_o}"
        )


def _install_one_requirement(py: str, line: str, *, label: str) -> None:
    if line.strip().startswith("onnx2tf"):
        _pip(py, [line, "--no-deps"], label=label)
        return

    if line.strip() == "webrtcvad":
        print("\n" + "=" * 72, flush=True)
        print(f"ETAP: {label} (PyPI → GitHub)", flush=True)
        print("=" * 72 + "\n", flush=True)
        last_err = ""
        for spec, sublabel in (
            (["webrtcvad"], "webrtcvad z PyPI"),
            (
                ["git+https://github.com/wiseman/py-webrtcvad.git"],
                "webrtcvad z GitHub",
            ),
        ):
            cmd = [py, "-m", "pip", "install", *spec]
            print(f"--- {sublabel} ---\n$ {' '.join(cmd)}\n", flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.stdout:
                print(r.stdout, end="", flush=True)
            if r.stderr:
                print(r.stderr, end="", file=sys.stderr, flush=True)
            if r.returncode == 0:
                return
            last_err = (r.stderr or "")[-6000:]
        raise RuntimeError(
            f"webrtcvad: PyPI i GitHub nie powiodły się.\n"
            f"ETAP: {label}\n\nOstatni stderr:\n{last_err}"
        )

    _pip(py, [line], label=label)


def _install_requirements_lines(py: str, req_path: Path, *, label_prefix: str) -> None:
    text = req_path.read_text(encoding="utf-8")
    n = 0
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        n += 1
        short = line if len(line) <= 72 else line[:69] + "…"
        _install_one_requirement(py, line, label=f"{label_prefix} [{n}] {short}")


def main() -> None:
    colab_dir = Path(__file__).resolve().parent
    train_req = colab_dir / "requirements-colab-train.txt"
    tflite_req = colab_dir / "requirements-colab-tflite.txt"
    py = sys.executable

    if not train_req.is_file():
        raise RuntimeError(f"Brak pliku: {train_req}")
    if not tflite_req.is_file():
        raise RuntimeError(f"Brak pliku: {tflite_req}")

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
    print("\n" + "=" * 72, flush=True)
    print("ETAP: PyTorch (CUDA 12.4 — typowy GPU Colab)", flush=True)
    print("$ " + " ".join(torch_cmd), flush=True)
    print("=" * 72 + "\n", flush=True)
    r = subprocess.run(torch_cmd)
    if r.returncode != 0:
        print("\n[INFO] cu124 nie zadziałał — instaluję torch+torchaudio z PyPI.\n", flush=True)
        _pip(py, ["torch", "torchaudio"], label="PyTorch (fallback PyPI)")

    _install_requirements_lines(py, train_req, label_prefix="Trening")

    _install_requirements_lines(py, tflite_req, label_prefix="TFLite")

    _pip(py, ["--force-reinstall", "numpy==1.26.4"], label="Ponownie numpy==1.26.4 (po TF)")

    print("\nGotowe. Interpreter:", py, flush=True)
    print("Test: import torch; print(torch.__version__, torch.cuda.is_available())", flush=True)


if __name__ == "__main__":
    main()
