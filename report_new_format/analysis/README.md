# Dataset Analysis

This folder contains the report-ready analysis for the Yamanishi08 dataset.
All figures are generated with Matplotlib using the non-interactive `Agg`
backend, so the script can run in terminal and notebook environments without a
display server.

## Run

From the repository root:

```powershell
python report/analysis/dataset_analysis.py
```

Custom paths and image resolution can be supplied when needed:

```powershell
python report/analysis/dataset_analysis.py `
  --data-dir data/yamanishi_08 `
  --output-dir report/analysis/results/yamanishi_08 `
  --dpi 300
```

## Outputs

The script recreates the `figures/` and `tables/` directories under the chosen
output directory and writes:

```text
report/analysis/results/yamanishi_08/
  analysis_report.md
  figures/
  tables/
```

Generated Matplotlib figures:

- `dti_degree_distribution.png`
- `dti_degree_rank.png`
- `network_sparsity_summary.png`
- `knowledge_graph_relation_distribution.png`
- `feature_characteristics.png`
- `warm_start_class_balance.png`
- `warm_start_fold_stability.png`

The PNG filenames are stable because they are referenced by the report source.
Use `--dpi` to control their rendering resolution.
