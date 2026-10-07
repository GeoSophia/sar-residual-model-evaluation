import unittest

import numpy as np
import pandas as pd

from sar_residual import paired_block_bootstrap, recovery_metrics, summarize_residuals
from sar_residual.demo import make_example


class AnalysisTests(unittest.TestCase):
    def test_summary_identity(self):
        result = summarize_residuals([[3, 4], [-3, -4]])
        self.assertEqual(result["range_rms_px"], 3)
        self.assertEqual(result["azimuth_population_sd_px"], 4)
        self.assertEqual(result["n"], 2)

    def test_empty_summary(self):
        result = summarize_residuals(np.empty((0, 2)))
        self.assertIsNone(result["range_rms_px"])
        self.assertEqual(result["n"], 0)

    def test_invalid_summary(self):
        for value in ([1, 2], [[1, np.nan]], [[1, 2, 3]]):
            with self.assertRaises(ValueError):
                summarize_residuals(value)

    def test_reproducible_demo(self):
        a = paired_block_bootstrap(make_example(), replicates=64)
        b = paired_block_bootstrap(make_example(), replicates=64)
        self.assertEqual(a, b)
        self.assertEqual(a["paired_n"], 288)
        self.assertEqual(a["blocks_n"], 36)

    def test_against_explicit_point_resampling(self):
        data = make_example()
        data = data.drop(data[(data.block_id == 0) & (data.point_id % 2 == 0)].index)
        count, seed = 75, 19
        expected = np.zeros((count, 5))
        rng = np.random.default_rng(seed)
        for _, frame in data.groupby("frame", sort=True):
            groups = [g for _, g in frame.groupby("block_id", sort=True)]
            draws = rng.integers(0, len(groups), size=(count, len(groups)))
            for i, draw in enumerate(draws):
                points = pd.concat([groups[j] for j in draw])
                expected[i] += [len(points)] + [(points[c]**2).sum() for c in
                                               ("range_a", "azimuth_a", "range_b", "azimuth_b")]
        result = paired_block_bootstrap(data, replicates=count, seed=seed)
        for axis, a, b in (("range", 1, 3), ("azimuth", 2, 4)):
            delta = np.sqrt(expected[:, b]/expected[:, 0]) - np.sqrt(expected[:, a]/expected[:, 0])
            row = next(r for r in result["components"] if r["component"] == axis)
            np.testing.assert_allclose([row["ci95_low_pixel"], row["ci95_high_pixel"]],
                                       np.quantile(delta, [.025, .975]), atol=1e-14)

    def test_identical_models(self):
        data = make_example()
        data["range_b"], data["azimuth_b"] = data.range_a, data.azimuth_a
        for row in paired_block_bootstrap(data, replicates=50)["components"]:
            self.assertEqual(row["delta_b_minus_a_pixel"], 0)
            self.assertEqual(row["ci95_low_pixel"], 0)
            self.assertEqual(row["ci95_high_pixel"], 0)

    def test_csv_numeric_strings(self):
        data = make_example()
        strings = data.astype({k: str for k in ["range_a", "azimuth_a", "range_b", "azimuth_b"]})
        self.assertEqual(paired_block_bootstrap(data, replicates=20),
                         paired_block_bootstrap(strings, replicates=20))

    def test_support_statuses(self):
        data = make_example()
        self.assertEqual(paired_block_bootstrap(data, replicates=10)["interval_status"],
                         "conditional_block_bootstrap")
        self.assertEqual(paired_block_bootstrap(data, replicates=10, declared_frames=[1,2,3,4])["interval_status"],
                         "incomplete_frame_support")
        self.assertEqual(paired_block_bootstrap(data[data.block_id < 2], replicates=10)["interval_status"],
                         "descriptive_sparse_blocks")
        self.assertEqual(paired_block_bootstrap(data.iloc[:0], replicates=10)["interval_status"],
                         "no_valid_samples")

    def test_reject_invalid_pairs(self):
        data = make_example()
        for bad in (data.drop(columns="range_a"), pd.concat([data, data.iloc[:1]])):
            with self.assertRaises(ValueError):
                paired_block_bootstrap(bad)
        data.loc[0, "range_a"] = np.inf
        with self.assertRaises(ValueError):
            paired_block_bootstrap(data)
        for kwargs in ({"replicates": 0}, {"replicates": True}, {"seed": -1}, {"seed": 1.2}):
            with self.assertRaises(ValueError):
                paired_block_bootstrap(make_example(), **kwargs)

    def test_recovery_denominators_and_gain(self):
        result = recovery_metrics([dict(valid=True, truth_range_px=2, recovered_range_px=1,
                                        bias_range_px=3), dict(valid=False)], "range")
        self.assertEqual(result["attempted_n"], 2)
        self.assertEqual(result["valid_n"], 1)
        self.assertEqual(result["recovery_gain"], .5)
        self.assertEqual(result["signed_bias_px"], -1)

    def test_zero_energy_and_no_valid(self):
        result = recovery_metrics([dict(valid=True, truth_range_px=0, recovered_range_px=1,
                                        bias_range_px=0)], "range")
        self.assertIsNone(result["recovery_gain"])
        self.assertIsNone(recovery_metrics([dict(valid=False)], "range")["rmse_px"])

    def test_reject_invalid_recovery(self):
        for records in ([dict(valid="False")], [dict(valid=True)],
                        [dict(valid=True, truth_range_px=np.nan, recovered_range_px=0, bias_range_px=0)]):
            with self.assertRaises(ValueError):
                recovery_metrics(records, "range")
        with self.assertRaises(ValueError):
            recovery_metrics([], "horizontal")


if __name__ == "__main__":
    unittest.main()
