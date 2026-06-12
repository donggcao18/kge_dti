# Dataset Analysis

This folder contains report-ready dataset analysis code for the DTI project.

## Run all analyses

From the repository root:

```powershell
python analysis/dataset_analysis.py --datasets yamanishi_08 BioKG
```

If the default `python` command is not working on this machine, use the bundled
Codex Python runtime:

```powershell
& "C:\Users\ADMIN\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" analysis\dataset_analysis.py --datasets yamanishi_08 BioKG
```

## Outputs

The script writes outputs to:

```text
analysis/results/
```

For each dataset, it creates:

```text
analysis/results/<dataset>/tables/
analysis/results/<dataset>/figures/
analysis/results/<dataset>/analysis_report.md
```

Important generated figures include:

- `degree_distribution.png` or `.svg`: drug and protein target degree distributions.
- `degree_ccdf_loglog.png` or `degree_rank_plot.svg`: heavy-tail analysis.
- `top_hubs.png` or separate `.svg` hub charts: top drug and target hubs.
- `kg_top_relations.png` or `.svg`: most frequent knowledge graph relation types.
- `split_class_balance.png` or `.svg`: train/test positive and negative class balance.
- `positive_entity_overlap.png` or `.svg`: cold-start and warm-start entity overlap.

If `matplotlib` is installed, the script writes PNG figures. If it is not
installed, the script falls back to dependency-light SVG charts.

Important generated tables include:

- `dti_network_summary.csv`
- `drug_degrees.csv`
- `target_degrees.csv`
- `top_25_drug_hubs.csv`
- `top_25_target_hubs.csv`
- `kg_summary.csv`
- `kg_relation_counts_combined.csv`
- `feature_summary.csv`
- `fold_level_split_stats.csv`
- `split_summary.csv`

The script also creates cross-dataset comparison files:

```text
analysis/results/dataset_comparison.csv
analysis/results/dataset_comparison.png
```
