"""Convert SavedModel directory to float32 TFLite (used after wrap_saved_model_wake_input)."""
from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("saved_model_dir", type=Path)
    parser.add_argument("output_tflite", type=Path)
    args = parser.parse_args()

    converter = tf.lite.TFLiteConverter.from_saved_model(str(args.saved_model_dir))
    converter.optimizations = []
    tflite_model = converter.convert()
    args.output_tflite.write_bytes(tflite_model)
    print(f"Wrote {args.output_tflite} ({len(tflite_model)} bytes)")


if __name__ == "__main__":
    main()
