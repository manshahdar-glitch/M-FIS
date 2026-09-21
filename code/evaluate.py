"""
Evaluation harness for M-FIS (reference implementation).

Reproduces the reported metrics from CACHED, L2-normalized foundation-model
embeddings: for each selection budget in {0.1, 0.2, 0.3} and each seed, it
selects a subset with M-FIS, trains the downstream classifiers, and reports
macro-F1 and macro-recall, together with the Wilcoxon signed-rank test of
M-FIS versus random sampling.

This is a minimal reference harness for the published method. The full
experimental framework (embedding extraction, dataset manifests, ablation
sweeps, and unpublished extensions) is withheld as it is part of ongoing
research. Provide your own cached embeddings via `load_cached_embeddings`.

Author: M. Manshah et al.  |  License: MIT
"""

from __future__ import annotations
import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score

from mfis_selector import GatedFusion, Probe, mfis_select

BUDGETS = (0.1, 0.2, 0.3)
SEEDS = (0, 1, 2, 3, 4, 5)          # six seeds, as in the paper


# ----------------------------------------------------------------------
# USER HOOK: return cached embeddings for one dataset.
# ----------------------------------------------------------------------
def load_cached_embeddings(dataset: str):
    """Return (z_img, z_txt, y) as float32 arrays.

    z_img, z_txt : (N, d) L2-normalized OpenCLIP ViT-B-32 (laion2b) embeddings
    y            : (N,) integer labels

    NOTE: embedding extraction/caching is intentionally not shipped in this
    reference release. Plug in your cached arrays here (e.g., np.load(...)).
    """
    raise NotImplementedError(
        "Provide cached OpenCLIP embeddings for '%s' here." % dataset)


# ----------------------------------------------------------------------
def train_fusion_probe(zv, zt, y, n_classes, epochs=18, seed=0, device="cpu"):
    """Train the gated fusion g and the probe f jointly (cross-entropy)."""
    torch.manual_seed(seed)
    d = zv.shape[1]
    fusion, probe = GatedFusion(d).to(device), Probe(d, n_classes).to(device)
    opt = torch.optim.Adam(list(fusion.parameters()) + list(probe.parameters()), lr=1e-3)
    Zv, Zt = torch.tensor(zv, device=device), torch.tensor(zt, device=device)
    Y = torch.tensor(y, dtype=torch.long, device=device)
    for _ in range(epochs):
        opt.zero_grad()
        p = probe(fusion(Zv, Zt))
        loss = torch.nn.functional.nll_loss(torch.log(p + 1e-12), Y)
        loss.backward(); opt.step()
    return fusion.eval(), probe.eval()


@torch.no_grad()
def probe_memberships(fusion, probe, zv, zt, device="cpu"):
    """Return (p, p_v, p_t): fused and the two modality-isolated predictions."""
    Zv, Zt = torch.tensor(zv, device=device), torch.tensor(zt, device=device)
    zero = torch.zeros_like(Zv)
    z = fusion(Zv, Zt)
    p  = probe(z).cpu().numpy()
    pv = probe(fusion(Zv, zero)).cpu().numpy()          # image only  f(g(z^v, 0))
    pt = probe(fusion(zero, Zt)).cpu().numpy()          # text only   f(g(0, z^t))
    return p, pv, pt, z.cpu().numpy()


def classifiers():
    return {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SVM":    LinearSVC(class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=200),
        "MLP":    MLPClassifier(hidden_layer_sizes=(256,), max_iter=300),
    }


def evaluate_dataset(dataset: str):
    zv, zt, y = load_cached_embeddings(dataset)
    n_classes = int(y.max()) + 1
    rows = []
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.30,
                                  stratify=y, random_state=seed)
        fusion, probe = train_fusion_probe(zv[tr], zt[tr], y[tr], n_classes, seed=seed)
        p, pv, pt, zf = probe_memberships(fusion, probe, zv, zt)
        for b in BUDGETS:
            # --- M-FIS selection on the training pool ---
            sel = mfis_select(zf[tr], y[tr], p[tr], pv[tr], pt[tr], rho=b, seed=seed)
            idx_ours = tr[sel]
            # --- random baseline at the same budget ---
            rng = np.random.default_rng(seed)
            idx_rand = tr[rng.choice(len(tr), size=len(sel), replace=False)]
            for name, clf in classifiers().items():
                for method, idx in [("M-FIS", idx_ours), ("Random", idx_rand)]:
                    m = clf.__class__(**clf.get_params())
                    m.fit(zf[idx], y[idx])
                    yp = m.predict(zf[te])
                    rows.append(dict(dataset=dataset, seed=seed, budget=b,
                                     classifier=name, method=method,
                                     f1=f1_score(y[te], yp, average="macro"),
                                     recall=recall_score(y[te], yp, average="macro")))
    return rows


def wilcoxon_report(rows, budget=0.2):
    """Wilcoxon signed-rank: M-FIS vs Random on macro-F1 (per classifier)."""
    import collections
    by = collections.defaultdict(lambda: {"M-FIS": [], "Random": []})
    for r in rows:
        if r["budget"] == budget:
            by[r["classifier"]][r["method"]].append(r["f1"])
    for clf, d in by.items():
        stat, p = wilcoxon(d["M-FIS"], d["Random"])
        print(f"{clf:14s} budget={budget}  "
              f"M-FIS={np.mean(d['M-FIS']):.3f}  "
              f"Random={np.mean(d['Random']):.3f}  Wilcoxon p={p:.3f}")


if __name__ == "__main__":
    for ds in ("crisismmd", "iuxray", "mmimdb"):
        try:
            rows = evaluate_dataset(ds)
            print(f"\n=== {ds} ===")
            wilcoxon_report(rows, budget=0.2)
        except NotImplementedError as e:
            print(f"[skip] {ds}: {e}")
