"""Synthetic usage example; no satellite data or manuscript result is implied."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import paired_block_bootstrap, recovery_metrics, summarize_residuals


def make_example():
    rng = np.random.default_rng(473110)
    rows = []
    for frame in (1, 2, 3):
        for block in range(12):
            common = rng.normal(0, 0.12, size=2)
            for point in range(8):
                a = common + rng.normal(0, 0.08, size=2)
                b = 0.85 * a + rng.normal(0, 0.03, size=2)
                rows.append(dict(frame=frame, block_id=block, point_id=block*8+point,
                                 range_a=a[0], azimuth_a=a[1], range_b=b[0], azimuth_b=b[1]))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New directory for synthetic CSV and JSON")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new directory")
    data = make_example()
    injections = [dict(valid=True, truth_range_px=1.0, recovered_range_px=0.9, bias_range_px=0.2),
                  dict(valid=True, truth_range_px=-1.0, recovered_range_px=-0.8, bias_range_px=0.1),
                  dict(valid=False)]
    result = dict(example_kind="synthetic_only_not_manuscript_results",
                  model_a=summarize_residuals(data[["range_a", "azimuth_a"]]),
                  comparison=paired_block_bootstrap(data, declared_frames=[1, 2, 3]),
                  injection=recovery_metrics(injections, "range"))
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        data.to_csv(args.output / "synthetic_paired_samples.csv", index=False)
        (args.output / "synthetic_summary.json").write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
