"""One-time conversion of trusted attachment-4 PKL files to ignored local NPZ files.

Pickle can execute code: use only competition files whose origin you already trust.
The application itself never accepts pickle uploads.
"""
import argparse
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trisense.features import SCHEMA, SHAPES, read_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trusted-pickle-dir", type=Path, required=True)
    parser.add_argument("--i-trust-these-files", action="store_true", required=True)
    args = parser.parse_args()
    import numpy.core as numpy_core
    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.numeric", numpy_core.numeric)
    sys.modules.setdefault("numpy._core.multiarray", numpy_core.multiarray)
    target = ROOT / "local_data"
    target.mkdir(exist_ok=True)
    count = 0
    for path in sorted(args.trusted_pickle_dir.glob("[0-9][0-9].pkl")):
        if not 1 <= int(path.stem) <= 20:
            continue
        with path.open("rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        result = target / f"S{path.stem}.npz"
        np.savez_compressed(result, schema=np.array(SCHEMA), synthetic=np.array(False),
                            **{key: np.asarray(data[key]) for key in SHAPES})
        read_features(result)
        count += 1
    if not count:
        raise SystemExit("No 01.pkl ... 20.pkl files found in the supplied directory.")
    print(f"Imported {count} private samples into ignored local_data/; no transcripts or videos copied.")


if __name__ == "__main__":
    main()
