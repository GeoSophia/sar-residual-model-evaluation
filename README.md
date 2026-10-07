# SAR Residual-Model Evaluation

Core statistical routines for evaluating residual models in DEM-assisted SAR
registration. This initial release contains code used for component-wise
residual summaries, paired spatial-block bootstrap comparisons, and
controlled-injection recovery metrics.

**Release scope:** this is a core-analysis release, not the complete processing
pipeline or a reproduction of all manuscript results. The example is synthetic;
it is not a Wenchuan observation or an independent validation dataset.

## Installation

Python 3.11 or newer is required. NumPy and pandas are installed as dependencies.

```sh
git clone https://github.com/GeoSophia/sar-residual-model-evaluation.git
cd sar-residual-model-evaluation
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install .
python -m unittest discover -s tests -v
python -m sar_residual.demo
```

To save the synthetic inputs and calculated summaries in a new directory:

```sh
python -m sar_residual.demo --output demo_results
```

The command refuses to overwrite an existing output directory. A typical run
uses 288 synthetic paired samples in 36 blocks across three frames and 5,000
bootstrap replicates. Runtime depends on the computer.

## Included Analyses

- `summarize_residuals(values)`: mean, population standard deviation, RMS,
  median, and fifth/ninety-fifth percentiles for range and azimuth residuals.
- `paired_block_bootstrap(table)`: resamples whole paired spatial blocks
  separately within each frame and calculates percentile intervals for RMS
  differences. Negative `delta_b_minus_a_pixel` means lower RMS for model B.
- `recovery_metrics(records, component)`: reports attempted/valid counts,
  recovery error, signed bias, and gain relative to known injected truth.

See [the data interface](docs/data_interface.md) for column names and examples.
The numerical kernel functions retain the original research implementation;
the public API adds explicit input validation. Source-function hashes and
origins are recorded in `SOURCE_PROVENANCE.json`.

## Interpretation

Intervals are conditional on the supplied paired measurements and fitted
predictions; the bootstrap does not refit models or correct for multiple
comparisons. All contributing frames should have adequate spatial-block
support. The API labels fewer than five blocks in any contributing frame as
descriptive sparse-block inference. This is an analysis convention, not a
general statistical guarantee.

Range and azimuth are native pixels with potentially different metre sampling.
The combined `2d` pixel statistic is not a physical displacement norm. Missing
or failed measurements must not be replaced with zeros. Zero-energy injection
cases have undefined recovery gain, represented by `None`/JSON `null`.

## Data and Further Development

Raw satellite imagery, DEMs, third-party software, local account information,
and unpublished manuscript files are not distributed here. Users must obtain
source products from their providers under the applicable terms. Additional
research materials may be requested from the corresponding author,
Zhen Li (`zhenli@bit.edu.cn`), subject to third-party restrictions.

Future releases are intended to add the image-processing entry points,
experiment configurations, numerical result checks, and figure-generation
workflows. Their inclusion is not claimed by this release. The repository is
not an assertion of acceptance by any journal.

## License

The source code in this release is provided under the MIT License. This license
does not apply to satellite products, DEMs, or other third-party materials.
