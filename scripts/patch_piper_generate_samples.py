"""Patch piper-sample-generator generate_samples.py: PyTorch>=2.6 wymaga weights_only=False dla .pt Pipera."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")
    if "torch.load(model_path, weights_only=False)" in text:
        return
    old = "model = torch.load(model_path)"
    if old not in text:
        raise SystemExit(
            f"patch_piper_generate_samples: brak linii {old!r} (inna wersja piper-sample-generator?)"
        )
    text = text.replace(
        old,
        "model = torch.load(model_path, weights_only=False)  # PyTorch>=2.6; checkpoint Piper (zaufane źródło)",
        1,
    )
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
