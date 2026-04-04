"""Uruchom onnx2tf z shimem onnx.helper.float32_to_bfloat16 (Colab / stary pakiet onnx).

Wywołanie jak CLI onnx2tf, np.:
  python scripts/run_onnx2tf_with_shim.py -i model.onnx -o out_dir -tb tf_converter
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    r = str(root)
    if r not in sys.path:
        sys.path.insert(0, r)

    from colab.onnx_helper_shim import apply_onnx_helper_bfloat16_shim

    apply_onnx_helper_bfloat16_shim()

    from onnx2tf.onnx2tf import main as onnx2tf_main

    sys.argv = ["onnx2tf", *sys.argv[1:]]
    onnx2tf_main()


if __name__ == "__main__":
    main()
