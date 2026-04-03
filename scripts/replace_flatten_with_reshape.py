"""Replace first Flatten with Reshape([1,1536]) for clearer shape inference in onnx2tf tf_converter."""
from __future__ import annotations

import argparse
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_onnx")
    parser.add_argument("output_onnx")
    args = parser.parse_args()

    model = onnx.load(args.input_onnx)
    g = model.graph

    flat = next((n for n in g.node if n.op_type == "Flatten" and n.name == "/flatten/Flatten"), None)
    if flat is None:
        raise SystemExit("Expected /flatten/Flatten node")

    x_in = flat.input[0]
    y_out = flat.output[0]

    shape_init = numpy_helper.from_array(
        np.array([1, 1536], dtype=np.int64),
        name="/flatten/reshape_shape",
    )
    reshape = helper.make_node(
        "Reshape",
        inputs=[x_in, shape_init.name],
        outputs=[y_out],
        name="/flatten/Reshape",
    )

    new_nodes = []
    replaced = False
    for n in g.node:
        if n.name == flat.name:
            new_nodes.append(reshape)
            replaced = True
        else:
            new_nodes.append(n)
    if not replaced:
        raise SystemExit("flatten not replaced")

    g.ClearField("node")
    g.node.extend(new_nodes)
    g.initializer.append(shape_init)

    onnx.checker.check_model(model)
    onnx.save(model, args.output_onnx)
    print(f"Wrote {args.output_onnx}")


if __name__ == "__main__":
    main()
