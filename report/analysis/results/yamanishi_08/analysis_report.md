# Yamanishi08 Dataset Analysis

This analysis is restricted to Yamanishi08 and the two warm-start settings used by the project. Unobserved drug-target pairs are described as *sampled negatives* rather than confirmed biological negatives.

## Dataset Overview

- Known positive interactions: 5,127
- Drugs: 791
- Target proteins: 989
- Possible drug-target pairs: 782,299
- Observed positive density: 0.66%
- Unobserved-pair sparsity: 99.34%
- Connected components: 57; largest component contains 81.29% of DTI nodes

## Degree Imbalance

- Median drug degree: 2.0; maximum: 132; Gini coefficient: 0.669
- Median target degree: 2.0; maximum: 61; Gini coefficient: 0.568
- Drugs with at most five interactions: 588 (74.34%)
- Targets with at most two interactions: 544 (55.01%)
- Top 10% of drugs account for 57.03% of interactions
- Top 10% of targets account for 41.49% of interactions

**Interpretation.** The graph combines extreme pair sparsity with concentrated hubs. Aggregate metrics can therefore be dominated by well-connected entities, while low-degree drugs and targets remain the harder cases.

## Knowledge Graph Context

- Raw KG rows: 95,672
- Unique KG triples: 95,579
- Duplicate KG triples removed: 93
- Entities: 25,487
- Relations: 48
- Directed entity-pair density: 0.01%
- Relation-aware density: 3.065e-06
- DTI drug coverage in the KG: 100.00%
- DTI target coverage in the KG: 100.00%

## Descriptor Quality

- Morgan fingerprints: 791 x 1024; zero share=95.86%; mean active bits=42.38
- Protein CTD descriptors: 989 x 147; zero share=0.00%; constant dimensions=0
- Missing or infinite values: 0 missing, 0 infinite

## Warm-Start Settings

### Warm-start 1:1

- Mean training pairs: 9,164.1 (4,615.0 positive, 4,549.1 sampled negative)
- Mean test pairs: 1,022.9 (512.0 positive, 510.9 sampled negative)
- Realized test negative-to-positive ratio: 0.998:1
- Minimum test-drug training coverage: 100.00%
- Minimum test-target training coverage: 100.00%

### Warm-start 1:10

- Mean training pairs: 44,864.2 (4,615.0 positive, 40,249.2 sampled negative)
- Mean test pairs: 5,538.9 (512.0 positive, 5,026.9 sampled negative)
- Realized test negative-to-positive ratio: 9.818:1
- Minimum test-drug training coverage: 100.00%
- Minimum test-target training coverage: 100.00%

## Data Quality Checks

- **Warm-start 1:1:** pair overlap=0, duplicate rows=0, negative/known-positive collisions=0, gold positives covered across test folds=60.85%.
- **Warm-start 1:10:** pair overlap=0, duplicate rows=0, negative/known-positive collisions=0, gold positives covered across test folds=60.85%.

The ten files are repeated warm-start holdouts rather than a disjoint ten-fold partition: test sets contain 3,120 unique positives in total, and some positive pairs occur in multiple test folds. Accordingly, results should be reported as the mean and standard deviation across repeated splits.

These checks are important because leakage, mislabeled sampled negatives, or incomplete warm-start entity coverage would make performance estimates difficult to interpret.

## Generated Figures

- `figures/dti_degree_distribution.png`
- `figures/dti_degree_rank.png`
- `figures/network_sparsity_summary.png`
- `figures/knowledge_graph_relation_distribution.png`
- `figures/feature_characteristics.png`
- `figures/warm_start_class_balance.png`
- `figures/warm_start_fold_stability.png`
