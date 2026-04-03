"""Strip broken in-container onnx-tf conversion from openWakeWord train.py (v0.6.x)."""
from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")
    before = text.count("convert_onnx_to_tflite(os.path.join")
    pattern = re.compile(
        r"convert_onnx_to_tflite\(os\.path\.join\(config\[\"output_dir\"\], "
        r'config\["model_name"\] \+ "\.onnx"\),\s*'
        r'os\.path\.join\(config\["output_dir"\], config\["model_name"\] \+ "\.tflite"\)\)',
        re.MULTILINE,
    )
    repl = 'print("Skipping ONNX->TFLite in container; ONNX export complete.")'
    text_new, n = pattern.subn(repl, text, count=1)
    if n != 1 or before < 1:
        raise SystemExit(
            f"patch_openwakeword_train: expected 1 replacement, got {n} "
            f"(call sites with os.path.join before: {before})"
        )
    path.write_text(text_new, encoding="utf-8")


if __name__ == "__main__":
    main()
