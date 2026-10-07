"""Validated interfaces to the original statistical kernels."""

import numbers

import numpy as np
import pandas as pd

from . import _kernels


def summarize_residuals(values):
    """Summarize an N-by-2 array in [range, azimuth] native pixels."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError("Expected a finite N-by-2 [range, azimuth] array")
    rows = [dict(residual_displacement_range_pixel=x,
                 residual_displacement_azimuth_pixel=y) for x, y in array]
    return _kernels.metrics(rows)


def paired_block_bootstrap(table, *, replicates=5000, seed=473110,
                           declared_frames=None):
    """Compare already-paired A/B observations using within-frame blocks.

    Rows must contain frame, block_id, range_a, azimuth_a, range_b, azimuth_b.
    Point pairing and exclusion of failed observations are caller responsibilities.
    """
    if isinstance(replicates, bool) or not isinstance(replicates, numbers.Integral) or replicates < 2:
        raise ValueError("replicates must be an integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    required = ["frame", "block_id", "range_a", "azimuth_a", "range_b", "azimuth_b"]
    data = pd.DataFrame(table).copy()
    if not set(required).issubset(data.columns):
        raise ValueError("Missing required columns: " + ", ".join(sorted(set(required) - set(data.columns))))
    if data[required].isna().any().any():
        raise ValueError("Paired inputs must not contain missing values")
    values = data[required[2:]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Residual components must be finite")
    for index, column in enumerate(required[2:]):
        data[column] = values[:, index]
    if "point_id" in data and data.point_id.isna().any():
        raise ValueError("point_id must not contain missing values")
    if "point_id" in data and data.duplicated(["frame", "point_id"]).any():
        raise ValueError("Duplicate point_id within a frame")
    actual = list(data.frame.unique())
    declared = list(declared_frames) if declared_frames is not None else actual
    if len(set(declared)) != len(declared):
        raise ValueError("declared_frames must not contain duplicates")
    if not set(actual).issubset(declared):
        raise ValueError("An observed frame is not in declared_frames")
    rows, blocks, fewest = _kernels.block_bootstrap(data, {"replicates": int(replicates)}, int(seed))
    if data.empty:
        status = "no_valid_samples"
    elif set(actual) != set(declared):
        status = "incomplete_frame_support"
    elif fewest < 5:
        status = "descriptive_sparse_blocks"
    else:
        status = "conditional_block_bootstrap"
    return {"paired_n": len(data), "blocks_n": blocks,
            "fewest_blocks_per_contributing_frame": fewest,
            "interval_status": status, "replicates": int(replicates),
            "seed": int(seed), "components": rows}


def recovery_metrics(records, component):
    """Summarize injected truth/recovery, retaining the attempted denominator."""
    if component not in ("range", "azimuth"):
        raise ValueError("component must be range or azimuth")
    records = list(records)
    for record in records:
        if not isinstance(record.get("valid"), (bool, np.bool_)):
            raise ValueError("valid must be a boolean, not a string or integer")
        if record["valid"]:
            keys = [f"{prefix}_{component}_px" for prefix in ("truth", "recovered", "bias")]
            if not all(key in record for key in keys):
                raise ValueError("Missing truth, recovered or bias component")
            if not np.isfinite([record[key] for key in keys]).all():
                raise ValueError("Valid records must have finite components")
    return _kernels.metric(records, component)
