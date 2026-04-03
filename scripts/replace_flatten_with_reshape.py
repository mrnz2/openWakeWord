"""Replace first Flatten with Reshape([1,N]) for onnx2tf tf_converter.

Stary eksporter PyTorch nazywał węzeł ``/flatten/Flatten``. Nowszy (onnxscript / PT 2.4+)
nadaje inne nazwy — szukamy pierwszego ``Flatten`` albo pomijamy, jeśli go nie ma
(wtedy kopiujemy model bez zmian).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def _pick_flatten(g: onnx.GraphProto) -> onnx.NodeProto | None:
    for n in g.node:
        if n.op_type == "Flatten" and n.name == "/flatten/Flatten":
            return n
    for n in g.node:
        if n.op_type == "Flatten":
            return n
    return None


def _dims_for_tensor_name(g: onnx.GraphProto, name: str) -> list[int] | None:
    for vi in list(g.value_info) + list(g.output) + list(g.input):
        if vi.name != name:
            continue
        tt = vi.type.tensor_type
        if not tt.shape.dim:
            continue
        dims: list[int] = []
        for d in tt.shape.dim:
            if d.dim_value:
                dims.append(int(d.dim_value))
            else:
                dims.append(-1)
        return dims
    return None


def _product_tail(dims: list[int]) -> int | None:
    """Default Flatten axis=1: product of all dims after batch."""
    if len(dims) < 2:
        return None
    p = 1
    for d in dims[1:]:
        if d <= 0:
            return None
        p *= d
    return p


def _flattened_dim_for_output(model: onnx.ModelProto, out_name: str, flat_in: str) -> int | None:
    try:
        inferred = shape_inference.infer_shapes(model)
        g = inferred.graph
    except Exception:
        g = model.graph
    out_dims = _dims_for_tensor_name(g, out_name)
    if out_dims is not None:
        if len(out_dims) >= 2 and out_dims[-1] > 0:
            return int(out_dims[-1])
        if len(out_dims) == 2 and out_dims[0] > 0 and out_dims[1] > 0:
            return int(out_dims[1])
    in_dims = _dims_for_tensor_name(g, flat_in)
    if in_dims is not None:
        return _product_tail(in_dims)
    return None


def _unique_name(existing: set[str], base: str) -> str:
    if base not in existing:
        return base
    for i in range(1, 10_000):
        cand = f"{base}_{i}"
        if cand not in existing:
            return cand
    raise RuntimeError(f"could not allocate unique name from {base!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_onnx")
    parser.add_argument("output_onnx")
    args = parser.parse_args()

    model = onnx.load(args.input_onnx)
    g = model.graph

    flat = _pick_flatten(g)
    if flat is None:
        print(
            "No Flatten node found (graph may already use Reshape); leaving model unchanged.",
            file=sys.stderr,
        )
        onnx.save(model, args.output_onnx)
        return

    x_in = flat.input[0]
    y_out = flat.output[0]

    n_feat = _flattened_dim_for_output(model, y_out, x_in)
    if n_feat is None:
        n_feat = 1536
        print(
            f"Could not infer Flatten output dim for {y_out!r}; using fallback N={n_feat}",
            file=sys.stderr,
        )

    existing_names = {i.name for i in g.initializer} | {n.name for n in g.node}
    shape_init_name = _unique_name(existing_names, "oww_flatten_reshape_shape")
    existing_names.add(shape_init_name)

    shape_init = numpy_helper.from_array(
        np.array([1, n_feat], dtype=np.int64),
        name=shape_init_name,
    )
    reshape_name = _unique_name(existing_names, "/flatten/Reshape")
    reshape = helper.make_node(
        "Reshape",
        inputs=[x_in, shape_init.name],
        outputs=[y_out],
        name=reshape_name,
    )

    new_nodes: list[onnx.NodeProto] = []
    for n in g.node:
        if n is flat:
            new_nodes.append(reshape)
        else:
            new_nodes.append(n)

    g.ClearField("node")
    g.node.extend(new_nodes)
    g.initializer.append(shape_init)

    onnx.checker.check_model(model)
    onnx.save(model, args.output_onnx)
    print(f"Wrote {args.output_onnx} (Flatten -> Reshape [1, {n_feat}])")


if __name__ == "__main__":
    main()
