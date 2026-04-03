import argparse
import re
import shlex
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


def run(cmd):
    print(f"\n$ {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def ensure_validation_features(project_dir: Path) -> Path:
    assets_dir = project_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_file = assets_dir / "validation_set_features.npy"
    if out_file.exists() and out_file.stat().st_size > 0:
        print(f"Validation features already present: {out_file}")
        return out_file

    url = (
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/"
        "resolve/main/validation_set_features.npy?download=true"
    )
    print(f"Downloading validation features (~185MB) to: {out_file}")
    urllib.request.urlretrieve(url, str(out_file))
    if not out_file.exists() or out_file.stat().st_size == 0:
        raise RuntimeError("Download finished but validation features file is missing/empty.")
    return out_file


def read_model_name(config_path: Path) -> str:
    text = config_path.read_text(encoding="utf-8")
    match = re.search(r'^\s*model_name\s*:\s*["\']?([^"\']+)["\']?\s*$', text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not read model_name from config: {config_path}")
    return match.group(1).strip()


def reset_workspace(project_dir: Path) -> None:
    """Remove outputs*, downloaded validation features, and Docker build context junk."""
    removed = []
    for p in sorted(project_dir.glob("outputs*")):
        if p.is_dir():
            shutil.rmtree(p)
            removed.append(str(p))
    assets = project_dir / "assets"
    if assets.is_dir():
        shutil.rmtree(assets)
        removed.append(str(assets))
    if removed:
        print("Removed:")
        for line in removed:
            print(f"  {line}")
    else:
        print("Nothing to remove (no outputs*/assets).")
    print(
        "Docker image wakeword-trainer:latest was not removed. "
        "To reclaim space: docker rmi wakeword-trainer:latest"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Runs real openWakeWord training in Docker and stores "
            "model artifacts in a local output directory."
        )
    )
    parser.add_argument(
        "--image",
        default="wakeword-trainer:latest",
        help="Docker image name used for build/run.",
    )
    parser.add_argument(
        "--training_config",
        default="training_configs/hey_lolita.yml",
        help="Relative path to openWakeWord training config YAML.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs_hey_lolita",
        help="Local directory for training artifacts.",
    )
    parser.add_argument(
        "--build_only",
        action="store_true",
        help="Build Docker image only and exit.",
    )
    parser.add_argument(
        "--shm_size",
        default="4g",
        help="Docker shared memory (--shm-size). On 16 GB RAM laptops 4g is safer; use 8g if stable.",
    )
    parser.add_argument(
        "--reset_workspace",
        action="store_true",
        help="Delete outputs* and assets/ then exit (fresh start; re-downloads validation on next run).",
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Skip clip generation and augmentation; run training stage only.",
    )
    parser.add_argument(
        "--force_overwrite",
        action="store_true",
        help="Force overwrite of intermediate files in openWakeWord pipeline.",
    )
    parser.add_argument(
        "--openwakeword_args",
        default="",
        help=(
            "Extra arguments passed directly to openwakeword/train.py in "
            'the container, e.g. --openwakeword_args "--overwrite".'
        ),
    )
    parser.add_argument(
        "--skip_tflite_conversion",
        action="store_true",
        help="Skip post-training ONNX->TFLite conversion step.",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    if args.reset_workspace:
        reset_workspace(project_dir)
        return

    output_dir = (project_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_validation_features(project_dir)
    config_path = (project_dir / args.training_config).resolve()
    if not config_path.exists():
        print(f"Training config not found: {config_path}", file=sys.stderr)
        sys.exit(2)
    if project_dir not in config_path.parents and config_path != project_dir:
        print("Training config must be inside the project directory.", file=sys.stderr)
        sys.exit(2)
    config_in_container = f"/workspace/project/{config_path.relative_to(project_dir).as_posix()}"
    model_name = read_model_name(config_path)

    print("--- OpenWakeWord Docker trainer ---")
    print(f"Project dir: {project_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Config file: {config_path}")

    run(["docker", "build", "-t", args.image, str(project_dir)])

    if args.build_only:
        print("\nImage built. Skipping training (--build_only).")
        return

    train_cmd = [
        "docker",
        "run",
        "--rm",
        "--shm-size",
        args.shm_size,
        "-v",
        f"{project_dir}:/workspace/project",
        "-v",
        f"{output_dir}:/workspace/outputs",
        args.image,
        "--training_config",
        config_in_container,
        "--train_model",
    ]
    if args.force_overwrite:
        train_cmd.append("--overwrite")
    if not args.train_only:
        train_cmd.extend(["--generate_clips", "--augment_clips"])
    if args.openwakeword_args.strip():
        train_cmd.extend(shlex.split(args.openwakeword_args))

    run(train_cmd)
    print("\nTraining finished.")

    if not args.skip_tflite_conversion:
        # 1) Last Gemm -> MatMul+Add (avoids onnx2tf ONNX_GEMM custom op in Wyoming).
        onnx_for_tflite = f"{model_name}_for_tflite.onnx"
        onnx_tfconv = f"{model_name}_tfconv.onnx"
        sm_tfconv_dir = f"onnx2tf_{model_name}_tfconv"
        wrapped_sm = f"wrapped_{model_name}_sm"
        final_tflite = str((output_dir / f"{model_name}.tflite").resolve())

        run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "python",
                "-v",
                f"{project_dir}:/workspace/project",
                "-v",
                f"{output_dir}:/workspace/outputs",
                args.image,
                "/workspace/project/scripts/rewrite_last_gemm_to_matmul.py",
                f"/workspace/outputs/{model_name}.onnx",
                f"/workspace/outputs/{onnx_for_tflite}",
            ]
        )
        # 2) Flatten -> Reshape so onnx2tf tf_converter infers [1,1536] correctly.
        run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "python",
                "-v",
                f"{project_dir}:/workspace/project",
                "-v",
                f"{output_dir}:/workspace/outputs",
                args.image,
                "/workspace/project/scripts/replace_flatten_with_reshape.py",
                f"/workspace/outputs/{onnx_for_tflite}",
                f"/workspace/outputs/{onnx_tfconv}",
            ]
        )
        # 3) ONNX -> SavedModel via tf_converter (writable copy avoids onnxsim permission errors).
        onnx2tf_shell = (
            f"cp /workdir/{onnx_tfconv} /tmp/in.onnx && "
            f"onnx2tf -i /tmp/in.onnx -o /workdir/{sm_tfconv_dir} -tb tf_converter"
        )
        run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "bash",
                "-v",
                f"{output_dir}:/workdir",
                "pinto0309/onnx2tf:latest",
                "-c",
                onnx2tf_shell,
            ]
        )
        # 4) Wrap: pyopen_wakeword feeds [1, N, 96]; inner SavedModel expects [1, 96, N].
        run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "python",
                "-v",
                f"{project_dir}:/workspace/project",
                "-v",
                f"{output_dir}:/workspace/outputs",
                args.image,
                "/workspace/project/scripts/wrap_saved_model_wake_input.py",
                f"/workspace/outputs/{sm_tfconv_dir}",
                f"/workspace/outputs/{wrapped_sm}",
            ]
        )
        # 5) Final float32 TFLite for Wyoming / Home Assistant.
        run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "python",
                "-v",
                f"{project_dir}:/workspace/project",
                "-v",
                f"{output_dir}:/workspace/outputs",
                args.image,
                "/workspace/project/scripts/export_saved_model_to_tflite.py",
                f"/workspace/outputs/{wrapped_sm}",
                f"/workspace/outputs/{model_name}.tflite",
            ]
        )
        print(f"\nSaved TFLite model: {final_tflite}")

    print("Check your model files in the output directory.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"Command failed with exit code {error.returncode}.", file=sys.stderr)
        sys.exit(error.returncode)