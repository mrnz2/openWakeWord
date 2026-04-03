"""
Wrap onnx2tf SavedModel so the TFLite signature accepts [1, 16, 96] (pyopen_wakeword /
Wyoming layout). Inner model expects [1, 96, 16] after onnx2tf NHWC transpose.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inner_saved_model_dir", type=Path)
    parser.add_argument("output_saved_model_dir", type=Path)
    args = parser.parse_args()

    inner = tf.saved_model.load(str(args.inner_saved_model_dir))
    fn = inner.signatures["serving_default"]
    # Typical key from onnx2tf: first arg name varies
    keys = list(fn.structured_input_signature[1].keys())
    if len(keys) != 1:
        raise SystemExit(f"Expected 1 input, got {keys}")
    in_key = keys[0]

    class Wrapped(tf.Module):
        def __init__(self, inner_fn, key: str):
            super().__init__()
            self._inner = inner_fn
            self._key = key

        @tf.function(
            input_signature=[tf.TensorSpec([1, 16, 96], tf.float32, name="wake_input")]
        )
        def serving_default(self, wake_input: tf.Tensor) -> tf.Tensor:
            x = tf.transpose(wake_input, [0, 2, 1])
            return self._inner(**{self._key: x})

    wrapped = Wrapped(fn, in_key)
    if args.output_saved_model_dir.exists():
        shutil.rmtree(args.output_saved_model_dir)
    tf.saved_model.save(
        wrapped,
        str(args.output_saved_model_dir),
        signatures={"serving_default": wrapped.serving_default.get_concrete_function()},
    )
    print(f"Saved wrapped model to {args.output_saved_model_dir}")


if __name__ == "__main__":
    main()
