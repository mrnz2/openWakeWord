"""Uzupełnia onnx.helper.float32_to_bfloat16, gdy w środowisku jest stary onnx (Colab / dist-packages).

onnx-graphsurgeon 0.5.x oczekuje tej funkcji przy imporcie; Debianowy python3-onnx jej nie ma.
Implementacja zgodna z ONNX 1.18 (_float32_to_bfloat16).
"""
from __future__ import annotations

import struct
from cmath import isnan


def _float32_to_bfloat16_impl(fval: float, truncate: bool = False) -> int:
    ival = int.from_bytes(struct.pack("<f", fval), "little")
    if truncate:
        return ival >> 16
    if isnan(fval):
        return 0x7FC0
    rounded = ((ival >> 16) & 1) + 0x7FFF
    return (ival + rounded) >> 16


def apply_onnx_helper_bfloat16_shim() -> bool:
    """Zwraca True, jeśli dołożono atrybut (albo już był)."""
    import onnx.helper as h

    if hasattr(h, "float32_to_bfloat16"):
        return True

    def float32_to_bfloat16(fval: float, truncate: bool = False) -> int:
        return _float32_to_bfloat16_impl(fval, truncate=truncate)

    h.float32_to_bfloat16 = float32_to_bfloat16  # type: ignore[attr-defined]
    return True
