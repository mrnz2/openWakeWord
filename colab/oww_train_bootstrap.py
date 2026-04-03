#!/usr/bin/env python3
"""Osobny proces treningu openWakeWord: usuwa debianowy site z sys.path przed importami (pkg_resources / Py 3.12)."""
import json
import os
import runpy
import sys


def _strip_debian_dist_paths() -> None:
    v = f"{sys.version_info.major}.{sys.version_info.minor}"
    local = f"/usr/local/lib/python{v}/dist-packages"
    keep: list[str] = []
    for p in sys.path:
        norm = p.replace("\\", "/")
        if p == "":
            keep.append(p)
            continue
        if norm.startswith("/usr/lib/") and norm.endswith("/dist-packages") and "/local/" not in norm:
            continue
        keep.append(p)
    sys.path[:] = keep
    if os.path.isdir(local):
        sys.path.insert(0, local)


def main() -> None:
    _strip_debian_dist_paths()
    root = os.environ["OWW_ROOT"]
    args = json.loads(os.environ["OWW_TRAIN_ARGV_JSON"])
    train_py = os.path.join(root, "openwakeword", "train.py")
    if not os.path.isfile(train_py):
        raise SystemExit(f"Brak pliku: {train_py}")
    os.chdir(root)
    sys.path.insert(0, root)
    sys.argv = [train_py] + args
    runpy.run_path(train_py, run_name="__main__")


if __name__ == "__main__":
    main()
