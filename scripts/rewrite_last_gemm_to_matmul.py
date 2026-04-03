"""
Replace final /last_layer/Gemm (ONNX Gemm with transB=1) by MatMul + Add so onnx2tf
can emit standard TFLite ops (no ONNX_GEMM custom op for Wyoming / stock TFLite).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_onnx")
    parser.add_argument("output_onnx")
    args = parser.parse_args()

    model = onnx.load(args.input_onnx)
    g = model.graph

    gemm = next((n for n in g.node if n.name == "/last_layer/Gemm"), None)
    if gemm is None:
        print("No /last_layer/Gemm found; leaving model unchanged.", file=sys.stderr)
        onnx.save(model, args.output_onnx)
        return

    relu_out, w_name, b_name = gemm.input[0], gemm.input[1], gemm.input[2]
    gemm_out = gemm.output[0]

    def get_init(name: str) -> np.ndarray:
        for t in g.initializer:
            if t.name == name:
                return numpy_helper.to_array(t)
        raise KeyError(name)

    W = get_init(w_name)
    Wt = np.transpose(W).astype(np.float32)
    mm_init = numpy_helper.from_array(Wt, name="last_layer_mm_weight")

    matmul_out = "/last_layer/matmul_out"
    matmul = helper.make_node(
        "MatMul",
        inputs=[relu_out, "last_layer_mm_weight"],
        outputs=[matmul_out],
        name="/last_layer/MatMul",
    )
    add = helper.make_node(
        "Add",
        inputs=[matmul_out, b_name],
        outputs=[gemm_out],
        name="/last_layer/Add_bias",
    )

    new_nodes: list = []
    for n in g.node:
        if n.name == "/last_layer/Gemm":
            new_nodes.extend([matmul, add])
        else:
            new_nodes.append(n)

    new_inits = [i for i in g.initializer if i.name != w_name]
    new_inits.append(mm_init)

    del g.node[:]
    g.node.extend(new_nodes)
    del g.initializer[:]
    g.initializer.extend(new_inits)

    onnx.checker.check_model(model)
    onnx.save(model, args.output_onnx)
    print(f"Wrote {args.output_onnx}")


if __name__ == "__main__":
    main()
