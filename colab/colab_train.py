"""
Uruchomienie treningu openWakeWord na Google Colab bez Dockera.

Wymaga: Linux, git, zależności z colab/install_colab_deps.py (train + tflite), pakietów apt (espeak-ng).
Trening openWakeWord w tym samym procesie (jak kernel Colab); torchmetrics>=1.4 — bez pkg_resources / Python 3.12.

Przykład:
  python colab/colab_train.py --project_root /content/WakeWordProject
"""
from __future__ import annotations

import argparse
import os
import re
import runpy
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


class CommandFailed(RuntimeError):
    """Polecenie podrzędne zwróciło kod != 0; komunikat zawiera koniec jego stdout/stderr."""

OWW_ROOT = Path("/content/openwakeword_v060")


def _usr_local_dist_packages() -> Path | None:
    """Katalog site-packages pip na Linux/Colab (torch, torchmetrics z pip przed katalogiem klonu)."""
    v = f"{sys.version_info.major}.{sys.version_info.minor}"
    p = Path(f"/usr/local/lib/python{v}/dist-packages")
    return p if p.is_dir() else None


PIPER_MODEL_URL = (
    "https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/"
    "en-us-libritts-high.pt"
)


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    tail_max_chars: int = 48000,
) -> None:
    print(f"\n$ {' '.join(cmd)}\n")
    full_env = {**os.environ, **env} if env else None
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chunks: list[str] = []
    total = 0
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            chunks.append(line)
            total += len(line)
            while total > tail_max_chars and chunks:
                total -= len(chunks.pop(0))
    finally:
        proc.stdout.close()
    code = proc.wait()
    if code != 0:
        tail = "".join(chunks)
        raise CommandFailed(
            f"Polecenie zakończone kodem {code}:\n  {' '.join(cmd)}\n\n"
            f"--- ostatnie ~{tail_max_chars} znaków wyjścia ---\n{tail[-tail_max_chars:]}"
        )


def ensure_validation_features(project_dir: Path) -> Path:
    assets_dir = project_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_file = assets_dir / "validation_set_features.npy"
    if out_file.exists() and out_file.stat().st_size > 0:
        print(f"Validation features OK: {out_file}")
        return out_file
    url = (
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/"
        "resolve/main/validation_set_features.npy?download=true"
    )
    print(f"Pobieranie validation_set_features.npy (~185 MB) -> {out_file}")
    urllib.request.urlretrieve(url, str(out_file))
    if not out_file.exists() or out_file.stat().st_size == 0:
        raise RuntimeError("Pobieranie validation features nie powiodło się.")
    return out_file


def read_model_name_from_config(text: str) -> str:
    match = re.search(
        r'^\s*model_name\s*:\s*["\']?([^"\']+)["\']?\s*$',
        text,
        flags=re.MULTILINE,
    )
    if not match:
        raise RuntimeError("Brak model_name w YAML.")
    return match.group(1).strip()


def is_openwakeword_patched(train_py: Path) -> bool:
    # Musi zgadzać się z tekstem w scripts/patch_openwakeword_train.py (TFLite, nie TFlite).
    return train_py.exists() and "Skipping ONNX->TFLite in container" in train_py.read_text(
        encoding="utf-8"
    )


def ensure_openwakeword_env(project_dir: Path) -> None:
    train_py = OWW_ROOT / "openwakeword" / "train.py"
    patch_script = project_dir / "scripts" / "patch_openwakeword_train.py"
    if not patch_script.exists():
        raise FileNotFoundError(f"Brak patcha: {patch_script}")

    if OWW_ROOT.exists() and train_py.exists() and is_openwakeword_patched(train_py):
        print(f"openWakeWord już przygotowany: {OWW_ROOT}")
    else:
        if OWW_ROOT.exists():
            shutil.rmtree(OWW_ROOT)
        OWW_ROOT.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "clone",
                "--branch",
                "v0.6.0",
                "--depth",
                "1",
                "https://github.com/dscripka/openWakeWord.git",
                str(OWW_ROOT),
            ]
        )
        psg = OWW_ROOT / "piper-sample-generator"
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/dscripka/piper-sample-generator.git",
                str(psg),
            ]
        )
        models_dir = psg / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        piper_pt = models_dir / "en-us-libritts-high.pt"
        if not piper_pt.exists() or piper_pt.stat().st_size == 0:
            print(f"Pobieranie modelu Piper -> {piper_pt}")
            urllib.request.urlretrieve(PIPER_MODEL_URL, str(piper_pt))

    # Idempotentny patch (TFLite + strażnik pustego positive_test) — także dla starego klonu w /content.
    run([sys.executable, str(patch_script), str(train_py)])

    # Embeddingi openWakeWord — wywołanie jest idempotentne (jak w Dockerfile).
    download_snippet = (
        "import os; "
        f"root={str(OWW_ROOT)!r}; "
        "os.chdir(root); "
        "import openwakeword.utils as u; "
        "u.download_models(target_directory=os.path.join(root,'openwakeword','resources','models'))"
    )
    run([sys.executable, "-c", download_snippet], cwd=OWW_ROOT)


def _replace_yaml_scalar_line(text: str, key: str, value_quoted: str, *, source: Path) -> str:
    """Zamienia jedną linię `key: ...` bez yaml.dump (openWakeWord ładuje config wrażliwym Loaderem)."""
    pattern = re.compile(rf"(?m)^{re.escape(key)}:\s*.+$")
    new_line = f'{key}: "{value_quoted}"'
    new_text, n = pattern.subn(new_line, text, count=1)
    if n != 1:
        raise RuntimeError(
            f"W pliku {source} nie udało się ustawić dokładnie jednej linii '{key}' (znaleziono {n})."
        )
    return new_text


def write_runtime_config(
    project_dir: Path,
    training_config_rel: str,
    output_dir: Path,
) -> tuple[Path, str]:
    src = project_dir / training_config_rel
    if not src.exists():
        raise FileNotFoundError(f"Brak configu: {src}")
    text = src.read_text(encoding="utf-8")
    model_name = read_model_name_from_config(text)
    out_abs = output_dir.resolve()
    val_abs = (project_dir / "assets" / "validation_set_features.npy").resolve()
    text = _replace_yaml_scalar_line(
        text, "output_dir", out_abs.as_posix(), source=src
    )
    text = _replace_yaml_scalar_line(
        text,
        "false_positive_validation_data_path",
        val_abs.as_posix(),
        source=src,
    )
    out = output_dir / "_colab_runtime_config.yml"
    output_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out, model_name


def run_openwakeword_train(
    config_in_container: Path,
    *,
    train_only: bool,
    force_overwrite: bool,
) -> None:
    train_py = OWW_ROOT / "openwakeword" / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(f"Brak {train_py} — uruchom setup (ensure_openwakeword_env).")

    argv_rest = [
        "--training_config",
        str(config_in_container),
        "--train_model",
    ]
    if force_overwrite:
        argv_rest.append("--overwrite")
    if not train_only:
        argv_rest.extend(["--generate_clips", "--augment_clips"])

    display_cmd = [sys.executable, "-m", "openwakeword.train", *argv_rest]
    print(f"\n$ {' '.join(display_cmd)}\n", flush=True)

    oww = str(OWW_ROOT.resolve())
    _lp = _usr_local_dist_packages()
    lp_s = str(_lp) if _lp else None
    while oww in sys.path:
        sys.path.remove(oww)
    if lp_s:
        while lp_s in sys.path:
            sys.path.remove(lp_s)
        sys.path.insert(0, oww)
        sys.path.insert(0, lp_s)
    else:
        sys.path.insert(0, oww)

    run_argv = [str(train_py), *argv_rest]
    old_argv = sys.argv[:]
    old_cwd = os.getcwd()
    try:
        os.chdir(OWW_ROOT)
        sys.argv = run_argv
        runpy.run_path(str(train_py), run_name="__main__")
    except SystemExit as e:
        code = e.code
        if code not in (0, None):
            raise CommandFailed(
                f"openwakeword.train zakończył się kodem {code!r}.\n"
                f"Polecenie (odpowiednik): {' '.join(display_cmd)}"
            ) from e
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


def run_tflite_pipeline(project_dir: Path, output_dir: Path, model_name: str) -> None:
    scripts = project_dir / "scripts"
    onnx_path = output_dir / f"{model_name}.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Brak ONNX: {onnx_path}")

    onnx_for_tflite = output_dir / f"{model_name}_for_tflite.onnx"
    onnx_tfconv = output_dir / f"{model_name}_tfconv.onnx"
    sm_tfconv_dir = output_dir / f"onnx2tf_{model_name}_tfconv"
    wrapped_sm = output_dir / f"wrapped_{model_name}_sm"
    final_tflite = output_dir / f"{model_name}.tflite"

    run(
        [
            sys.executable,
            str(scripts / "rewrite_last_gemm_to_matmul.py"),
            str(onnx_path),
            str(onnx_for_tflite),
        ]
    )
    run(
        [
            sys.executable,
            str(scripts / "replace_flatten_with_reshape.py"),
            str(onnx_for_tflite),
            str(onnx_tfconv),
        ]
    )

    tmp_in = Path("/tmp/colab_onnx2tf_in.onnx")
    shutil.copy(onnx_tfconv, tmp_in)
    if sm_tfconv_dir.exists():
        shutil.rmtree(sm_tfconv_dir)
    onnx2tf_bin = shutil.which("onnx2tf")
    if not onnx2tf_bin:
        raise RuntimeError("Brak polecenia onnx2tf w PATH (zainstaluj colab/requirements-colab.txt).")
    run(
        [
            onnx2tf_bin,
            "-i",
            str(tmp_in),
            "-o",
            str(sm_tfconv_dir),
            "-tb",
            "tf_converter",
        ]
    )

    run(
        [
            sys.executable,
            str(scripts / "wrap_saved_model_wake_input.py"),
            str(sm_tfconv_dir),
            str(wrapped_sm),
        ]
    )
    run(
        [
            sys.executable,
            str(scripts / "export_saved_model_to_tflite.py"),
            str(wrapped_sm),
            str(final_tflite),
        ]
    )
    print(f"\nZapisano TFLite: {final_tflite}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trening openWakeWord na Colab (bez Dockera).")
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path("/content/WakeWordProject"),
        help="Katalog z repozytorium WakeWordProject (skrypty, training_configs).",
    )
    parser.add_argument(
        "--training_config",
        default="training_configs/hey_lolita_colab.yml",
        help="Ścieżka względna od project_root.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/content/wakeword_outputs"),
        help="Katalog na clipy, ONNX i TFLite.",
    )
    parser.add_argument(
        "--skip_setup",
        action="store_true",
        help="Pomiń klonowanie openWakeWord / Piper (gdy już jest w /content/openwakeword_v060).",
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Tylko trening (zakłada gotowe clipy w output_dir).",
    )
    parser.add_argument(
        "--force_overwrite",
        action="store_true",
        help="Przekaż --overwrite do openwakeword.train.",
    )
    parser.add_argument(
        "--skip_tflite",
        action="store_true",
        help="Zakończ na ONNX (bez konwersji TFLite).",
    )
    args = parser.parse_args()

    project_dir = args.project_root.resolve()
    if not project_dir.is_dir():
        raise SystemExit(f"project_root nie istnieje: {project_dir}")

    if not args.skip_setup:
        ensure_validation_features(project_dir)
        ensure_openwakeword_env(project_dir)

    runtime_cfg, model_name = write_runtime_config(
        project_dir,
        args.training_config,
        args.output_dir.resolve(),
    )
    print(f"Runtime config: {runtime_cfg}")
    print(f"model_name: {model_name}")

    run_openwakeword_train(
        runtime_cfg,
        train_only=args.train_only,
        force_overwrite=args.force_overwrite,
    )

    if not args.skip_tflite:
        run_tflite_pipeline(project_dir, args.output_dir.resolve(), model_name)

    print("\nGotowe. Pobierz pliki z output_dir (np. zip albo skopiuj na Drive).")


if __name__ == "__main__":
    try:
        main()
    except CommandFailed as exc:
        print(f"\n[colab_train] {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
