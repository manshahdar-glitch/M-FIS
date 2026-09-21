# M-FIS: Multimodal Fuzziness-based Instance Selection

Reproducibility repository for the paper
**"Fuzziness-based instance selection for data-efficient adaptation of multimodal
foundation models in clinical and crisis screening."**

M-FIS selects a compact, class-balanced subset of paired image–text samples for the
data-efficient adaptation of a frozen multimodal foundation model. It scores each sample
by combining entropy, margin, and boundary fuzziness with a cross-modal disagreement
measure (Jensen–Shannon divergence between image-only and text-only predictions),
categorizes samples with a Gaussian mixture, keeps the informative medium-fuzziness band,
and diversifies + class-rebalances the retained set.

## What is in this repository

- `results/` — the complete per-seed numerical results behind every table and figure,
  as `.xlsx` (raw runs + mean/std summary + significance sheets), for CrisisMMD, IU X-ray,
  and MM-IMDb. Four files per dataset: `*_results`, `*_rebalance`, `*_crossmodal`,
  `*_noise_results`.
- `figures/` — the main-paper figures.
- `code/` — a reference implementation of the M-FIS selector and the evaluation harness
  (see `code/README.md`).

## Environment

- Python 3.10+
- `pip install -r requirements.txt`
- Foundation model: frozen **OpenCLIP `ViT-B-32`, `laion2b_s34b_b79k`** (trained on LAION-2B).
  Embeddings are L2-normalized and cached once, then reused across all experiments.

## Datasets (public)

- **MM-IMDb** — poster image + plot synopsis (10 genres)
- **CrisisMMD** — tweet image + text (7 humanitarian categories)
- **IU X-ray** — chest radiograph + report (binary normal/abnormal via a negation-aware
  rule over the R2Gen release)

## Reproducing the results

- Metrics reported in the paper come from the evaluation harness at selection **budgets
  0.1 / 0.2 / 0.3** (10/20/30% of the training data), over **6 seeds**, with the
  **Wilcoxon signed-rank** test.
- Note: the exploratory `select_ratio` argument (0.71–0.89 per dataset) only controls the
  qualitative snapshot used for the PCA / example figures; it does **not** set the reported
  budgets, which are fixed at 0.1/0.2/0.3 in the evaluation function.

| Paper item | Results file |
|---|---|
| Table 3 / Fig. 4 (accuracy vs budget) | `results/<dataset>/<dataset>_results.xlsx` |
| Table 4 (balanced recall)             | `results/<dataset>/<dataset>_results.xlsx` |
| Table 5 / Fig. 5 (rebalance λ)         | `results/<dataset>/<dataset>_rebalance.xlsx` |
| Table 6 / Fig. 6 (cross-modal w₃)      | `results/<dataset>/<dataset>_crossmodal.xlsx` |
| Label-noise robustness (supplementary) | `results/<dataset>/<dataset>_noise_results.xlsx` |

## Citation

If you use this repository, please cite the paper (see `CITATION.cff`).

## License

Released under the MIT License (see `LICENSE`).

DOI: 10.5281/zenodo.22875130
