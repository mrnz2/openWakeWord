"""
Convert openWakeWord ONNX to TFLite. For Wyoming compatibility, pass ONNX that was
already processed with rewrite_last_gemm_to_matmul.py (avoids ONNX_GEMM custom op).
"""
import argparse
import glob
import shutil
import subprocess
import sys
from pathlib import Path

from onnx2tf import convert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_onnx", required=True)
    parser.add_argument("--output_tflite", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument(
        "--rewrite_last_gemm",
        action="store_true",
        help="Run scripts/rewrite_last_gemm_to_matmul.py into work_dir before onnx2tf.",
    )
    args = parser.parse_args()

    onnx_in = Path(args.input_onnx).resolve()
    if args.rewrite_last_gemm:
        script = Path(__file__).resolve().parent / "rewrite_last_gemm_to_matmul.py"
        fixed = Path(args.work_dir).resolve() / (onnx_in.stem + "_for_tflite.onnx")
        fixed.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, str(script), str(onnx_in), str(fixed)],
            check=True,
        )
        onnx_in = fixed

    convert(input_onnx_file_path=str(onnx_in), output_folder_path=args.work_dir)
    candidates = glob.glob(f"{args.work_dir}/**/*float32*.tflite", recursive=True)
    if not candidates:
        candidates = glob.glob(f"{args.work_dir}/**/*.tflite", recursive=True)
    if not candidates:
        raise RuntimeError("onnx2tf did not produce any .tflite file")
    candidates.sort()
    shutil.copyfile(candidates[0], args.output_tflite)
    print(f"Saved: {args.output_tflite}")


if __name__ == "__main__":
    main()
