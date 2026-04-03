#!/usr/bin/env python3
"""
Instalacja zależności pod Google Colab w bezpiecznej kolejności.

- Nie instaluje `openwakeword` z PyPI (klon v0.6.0 jest w colab_train.py).
- Każda linia z requirements = osobne `pip install` — widać winnego pakietu.
- Uruchamiaj z notebooka przez runpy.run_path(..., run_name="__main__"), żeby log był w tej samej komórce.
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
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(
            f"\n*** BŁĄD: etap „{label}” zakończył się kodem {r.returncode}. ***\n"
            "Jeśli to kompilacja (webrtcvad): apt musi mieć build-essential i python3-dev.\n",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(r.returncode)


def _install_one_requirement(py: str, line: str, *, label: str) -> None:
    """Instalacja jednej linii z requirements; zna obejścia dla problematycznych pakietów."""
    if line.strip().startswith("onnx2tf"):
        # Metadane onnx2tf wymagają pakietu „tensorflow”; pip doinstalowałby pełny TF obok tensorflow-cpu.
        _pip(py, [line, "--no-deps"], label=label)
        return

    if line.strip() == "webrtcvad":
        print("\n" + "=" * 72, flush=True)
        print(f"ETAP: {label} (próba PyPI, potem źródło z GitHub)", flush=True)
        print("=" * 72 + "\n", flush=True)
        for spec, sublabel in (
            (["webrtcvad"], "webrtcvad z PyPI"),
            (
                ["git+https://github.com/wiseman/py-webrtcvad.git"],
                "webrtcvad z github.com/wiseman/py-webrtcvad",
            ),
        ):
            cmd = [py, "-m", "pip", "install", *spec]
            print(f"--- {sublabel} ---\n$ {' '.join(cmd)}\n", flush=True)
            r = subprocess.run(cmd)
            if r.returncode == 0:
                return
        print("*** webrtcvad: obie próby nie powiodły się. ***", file=sys.stderr, flush=True)
        raise SystemExit(1)

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
        _install_one_requirement(
            py, line, label=f"{label_prefix} [{n}] {short}"
        )


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
