"""Original numerical functions; see SOURCE_PROVENANCE.json."""

import math
import numpy as np
import pandas as pd

COMPONENTS = ["range", "azimuth"]

def block_bootstrap(paired, settings, seed):
    """Resample whole paired test blocks within each frame using sufficient sums."""
    blocks = []
    for (frame, block), group in paired.groupby(["frame", "block_id"], sort=True):
        record = {"frame": frame, "block_id": block, "n": len(group)}
        for method in ("a", "b"):
            for component in COMPONENTS:
                x = group[f"{component}_{method}"].to_numpy(float)
                record[f"sum_{component}_{method}"] = x.sum()
                record[f"squares_{component}_{method}"] = (x*x).sum()
        blocks.append(record)
    b = pd.DataFrame(blocks)
    if b.empty:
        return [], 0, 0
    nrep = settings["replicates"]
    columns = ["n"]+[f"{stat}_{c}_{m}" for m in ("a", "b") for c in COMPONENTS for stat in ("sum", "squares")]
    rng = np.random.default_rng(seed)
    totals = np.zeros((nrep, len(columns)))
    fewest = min(b.groupby("frame").size())
    for _, frame in b.groupby("frame", sort=True):
        matrix = frame[columns].to_numpy(float)
        draws = rng.integers(0, len(matrix), size=(nrep, len(matrix)))
        totals += matrix[draws].sum(axis=1)
    boot = {name: totals[:, i] for i, name in enumerate(columns)}
    results = []
    for component in [*COMPONENTS, "2d"]:
        if component == "2d":
            mse_a = (boot["squares_range_a"]+boot["squares_azimuth_a"])/boot["n"]
            mse_b = (boot["squares_range_b"]+boot["squares_azimuth_b"])/boot["n"]
            obs_a = np.sqrt(np.mean(paired.range_a**2+paired.azimuth_a**2))
            obs_b = np.sqrt(np.mean(paired.range_b**2+paired.azimuth_b**2))
        else:
            mse_a = boot[f"squares_{component}_a"]/boot["n"]
            mse_b = boot[f"squares_{component}_b"]/boot["n"]
            obs_a = np.sqrt(np.mean(paired[f"{component}_a"]**2))
            obs_b = np.sqrt(np.mean(paired[f"{component}_b"]**2))
        delta = np.sqrt(mse_b)-np.sqrt(mse_a)
        lo, hi = np.quantile(delta, [.025, .975])
        a_low,a_high = np.quantile(np.sqrt(mse_a),[.025,.975])
        b_low,b_high = np.quantile(np.sqrt(mse_b),[.025,.975])
        results.append({"component": component, "rmse_a_pixel": float(obs_a),
                        "rmse_b_pixel": float(obs_b), "delta_b_minus_a_pixel": float(obs_b-obs_a),
                        "ci95_low_pixel": float(lo), "ci95_high_pixel": float(hi),
                        "rmse_a_ci95_low_pixel":float(a_low), "rmse_a_ci95_high_pixel":float(a_high),
                        "rmse_b_ci95_low_pixel":float(b_low), "rmse_b_ci95_high_pixel":float(b_high),
                        "relative_reduction_percent": float(100*(obs_a-obs_b)/obs_a) if obs_a else None})
    return results, len(b), int(fewest)

def metrics(rows):
    values = {"n": len(rows)}
    for axis in ("range", "azimuth"):
        data = np.asarray([q[f"residual_displacement_{axis}_pixel"] for q in rows], float)
        if not len(data):
            stats = {key: None for key in ("mean_px", "population_sd_px", "rms_px", "median_px", "p05_px", "p95_px")}
        else:
            mean, sd, rms = float(data.mean()), float(data.std(ddof=0)), float(np.sqrt(np.mean(data**2)))
            assert np.isclose(rms*rms, mean*mean+sd*sd, rtol=1e-12, atol=1e-12)
            stats = {"mean_px": mean, "population_sd_px": sd, "rms_px": rms, "median_px": float(np.median(data)),
                     "p05_px": float(np.quantile(data, .05)), "p95_px": float(np.quantile(data, .95))}
        values.update({f"{axis}_{key}": value for key, value in stats.items()})
    return values

def metric(records, component):
    valid = [q for q in records if q["valid"]]
    n = len(valid)
    truth = [q[f"truth_{component}_px"] for q in valid]
    observed = [q[f"recovered_{component}_px"] for q in valid]
    error = [y-x for x, y in zip(truth, observed)]
    energy = math.fsum(x*x for x in truth)
    return {"attempted_n": len(records), "valid_n": n,
        "rmse_px": math.sqrt(math.fsum(x*x for x in error)/n) if n else None,
        "signed_bias_px": math.fsum(error)/n if n else None,
        "recovery_gain": math.fsum(x*y for x, y in zip(truth, observed))/energy if energy > 0 else None,
        "truth_rms_px": math.sqrt(energy/n) if n else None,
        "sampled_bias_rms_px": math.sqrt(math.fsum(q[f"bias_{component}_px"]**2 for q in valid)/n) if n else None}
