#!/usr/bin/env python3
"""
Instalacja zależności pod Google Colab w bezpiecznej kolejności.

- Nie instaluje `openwakeword` z PyPI (klon v0.6.0 jest w colab_train.py).
- Każda linia z requirements = osobne `pip install` — widać winnego pakietu.
- Przy błędzie: RuntimeError z ETAP + końcówka stderr/stdout pip (w tracebacku na dole komórki).
- setuptools 69–81: pkg_resources (pronouncing itd.); setuptools>=82 **usuwa** pkg_resources.
- lightning-utilities>=0.12 przed Trening: torchmetrics nie może zostawiać starej wersji z pkg_resources.
- jedi>=0.16: razem z pip/wheel — ipython w Colab bez jedi = ostrzeżenie po setuptools.
- Po PyTorch: odinstalowanie `tensorflow` z Colab (2.19), żeby nie psuł protobuf/tensorboard przed Treningiem.
- Po numpy: ponownie tensorflow + utrwalenie `protobuf` 5.29+ (pip czasem zostawia 4.x po TF-cpu).
- Usuwane `dopamine-rl` (Colab): w metadanych wymaga `tensorflow`, którego nie trzymamy (jest `tensorflow-cpu`).
- Usuwane `tensorflow-decision-forests` (Colab): wymaga `tensorflow==2.19.0` — konflikt z `tensorflow-cpu` 2.18.
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


def _pip_try_uninstall(py: str, package: str, *, label: str) -> None:
    """pip uninstall -y (brak pakietu = OK)."""
    print("\n" + "=" * 72, flush=True)
    print(f"ETAP: {label}", flush=True)
    cmd = [py, "-m", "pip", "uninstall", "-y", package]
    print("$ " + " ".join(cmd), flush=True)
    print("=" * 72 + "\n", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout, end="", flush=True)
    if r.stderr:
        print(r.stderr, end="", file=sys.stderr, flush=True)


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

    # jedi w tej samej transakcji co pip — inaczej łatwo pominąć osobny ETAP przy starym sklonowanym repo.
    _pip(
        py,
        ["-U", "pip", "wheel", "jedi>=0.16"],
        label="Narzędzia pip + jedi (ipython w Colab)",
    )
    # Debian/Colab: /usr/lib/.../pkg_resources odwołuje się do pkgutil.ImpImporter (usunięty w 3.12).
    _pip(
        py,
        ["--force-reinstall", "setuptools>=69.2.0,<82"],
        label="setuptools 69–81 (pkg_resources; v82+ go nie dostarcza)",
    )

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

    # Zanim Trening ustawi protobuf — usuwamy pełny TF 2.19 z obrazu (kłóci się z TF-cpu 2.18 i tensorboard).
    _pip_try_uninstall(
        py,
        "tensorflow",
        label="Colab: odinstalowanie tensorflow (przed Trening — protobuf / tensorboard)",
    )
    _pip_try_uninstall(
        py,
        "dopamine-rl",
        label="Colab: dopamine-rl (wymaga pakietu tensorflow — niekompatybilne z tensorflow-cpu)",
    )
    _pip_try_uninstall(
        py,
        "tensorflow-decision-forests",
        label="Colab: tensorflow-decision-forests (wymaga tensorflow==2.19 — nieużywane z TF-cpu)",
    )

    _pip(
        py,
        ["lightning-utilities>=0.12.0"],
        label="lightning-utilities>=0.12 (importlib.metadata — bez pkg_resources / Py 3.12)",
    )

    _install_requirements_lines(py, train_req, label_prefix="Trening")

    _install_requirements_lines(py, tflite_req, label_prefix="TFLite")

    # Po TF: utrwal numpy 2.0.x (wymóg TF 2.18: <2.1; Colab ma pakiety wymagające numpy>=2).
    _pip(
        py,
        ["--force-reinstall", "numpy>=2.0.0,<2.1.0"],
        label="Ponownie numpy 2.0.x (po tensorflow-cpu, zgodność z Colab)",
    )

    _pip_try_uninstall(
        py,
        "tensorflow",
        label="Colab: tensorflow po całości (gdyby wrócił z metadanych innego pakietu)",
    )
    _pip_try_uninstall(
        py,
        "dopamine-rl",
        label="Colab: dopamine-rl po całości (fałszywy brak tensorflow przy protobuf)",
    )
    _pip_try_uninstall(
        py,
        "tensorflow-decision-forests",
        label="Colab: tensorflow-decision-forests po całości (ostrzeżenie pip przy numpy)",
    )
    _pip(
        py,
        ["--upgrade", "--force-reinstall", "protobuf>=5.29.1,<6"],
        label="protobuf 5.29+ (utrwalenie — ydf/grain vs stary 4.x po pip)",
    )

    print(
        "\n[INFO] Pojedyncze „dependency conflicts” mogą dotyczyć paczek Colab poza tym notebookiem; "
        "sukces = „Successfully installed …” i brak kodu błędu pip.\n",
        flush=True,
    )
    print("\nGotowe. Interpreter:", py, flush=True)
    print("Test: import torch; print(torch.__version__, torch.cuda.is_available())", flush=True)


if __name__ == "__main__":
    main()
