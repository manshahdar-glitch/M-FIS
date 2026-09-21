# Code

Place here the reference implementation of M-FIS:

- `mfis_selector.py` — the fuzziness scores (entropy, margin, boundary), the cross-modal
  disagreement (normalized JS divergence), the GMM (BIC) categorization, medium-band
  selection, farthest-point diversification, and the class-rebalanced budget.
- `evaluate.py` — the evaluation harness that reproduces the tables from cached embeddings
  at budgets 0.1 / 0.2 / 0.3 over 6 seeds with the Wilcoxon test.

This is a clean reference implementation of the published method only; the full data
pipeline and unpublished extensions are intentionally not included and will be published after paper publicatio or on demand.
