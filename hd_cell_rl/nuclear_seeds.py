"""One confident nuclear-seed definition for inference and evaluation."""
from pathlib import Path
import warnings

import pandas as pd
from pyarrow import ArrowInvalid


def load_confident_nuclear_rows(path: Path) -> pd.DataFrame:
    columns = ["barcode", "dominant_cell_id", "has_nuclear_annotation"]
    try:
        frame = pd.read_parquet(path, columns=columns + ["ambiguous_nuclear_assignment"])
    except (KeyError, ArrowInvalid):
        # Historical metadata predates ambiguity annotations. Keep it readable,
        # but do not silently give it the same provenance as current seed data.
        frame = pd.read_parquet(path, columns=columns)
        warnings.warn("Legacy nuclear metadata lacks ambiguity flags; treating annotated bins as confident", RuntimeWarning)
    annotated = frame["has_nuclear_annotation"].fillna(False).astype(bool)
    ambiguous = frame.get("ambiguous_nuclear_assignment", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    return frame.loc[annotated & ~ambiguous].copy()
