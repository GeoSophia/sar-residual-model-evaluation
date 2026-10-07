# Data Interface

## Residual Summary

```python
from sar_residual import summarize_residuals

summary = summarize_residuals([[0.10, -0.20], [-0.05, 0.15]])
```

Each row is `[range, azimuth]` in native pixels. The array must have shape
`(N, 2)` and contain only finite values. Empty `(0, 2)` input returns `n=0`
and undefined component statistics.

## Paired Spatial-Block Bootstrap

```python
import pandas as pd
from sar_residual import paired_block_bootstrap

pairs = pd.read_csv("paired_observations.csv")
result = paired_block_bootstrap(pairs, replicates=5000, seed=473110,
                                declared_frames=[610, 620, 630])
```

Required columns:

| Column | Meaning |
| --- | --- |
| `frame` | Frame/scene stratum identifier |
| `block_id` | Spatial-block identifier, unique within each frame |
| `range_a`, `azimuth_a` | Model A residual components in native pixels |
| `range_b`, `azimuth_b` | Model B residuals for the same observation |
| `point_id` | Optional observation identifier, unique within a frame |

Pair observations by their identity before calling the function. Supply only
observations valid for both models and separately retain/report the full
attempted denominator and failure reasons. The function does not construct
training/test splits, check spatial separation, fit models, determine block
size, or establish a stable reference area. These choices require the full
experiment protocol. Frame and block identifiers must have consistent types.

Within each frame, each replicate samples the original number of blocks with
replacement. All points in a sampled block travel together. Component RMS is
calculated from pooled sums of squares and counts, then B-minus-A differences
are summarized by the 2.5th and 97.5th percentiles. Repeated blocks contribute
repeated points; observations are not resampled independently.

The returned status is `no_valid_samples`, `incomplete_frame_support`,
`descriptive_sparse_blocks`, or `conditional_block_bootstrap`. Empty input
returns no component intervals, not fabricated zero errors. Supplying
`declared_frames` allows omitted frames to be detected.

## Injection Metrics

```python
from sar_residual import recovery_metrics

rows = [
    {"valid": True, "truth_range_px": 1.0,
     "recovered_range_px": 0.8, "bias_range_px": 0.2},
    {"valid": False},
]
summary = recovery_metrics(rows, "range")
```

`valid` must be a real boolean. For valid rows, `truth`, `recovered` and `bias`
values for the selected component are mandatory. `bias` is the known sampled
injected bias, not the recovery error. Errors are recovered minus truth; gain
is `sum(truth * recovered) / sum(truth**2)`. Invalid rows remain in
`attempted_n` but not `valid_n`. Zero truth energy yields an undefined gain.

## Scientific Boundary

These routines evaluate supplied measurements. They cannot establish absolute
ground truth, universal algorithm superiority, or independence of observations.
The synthetic example tests execution only. A complete paper-result replay
requires the relevant measurements, configurations and remaining processing
modules, which are not included in this initial release.
