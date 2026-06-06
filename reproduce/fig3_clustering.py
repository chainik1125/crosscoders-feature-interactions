"""Reproduce Figure 3: clustering crosscoder features by interaction metric, and
the GPT-4o cluster-assignment accuracy.

Affinity propagation is run on the (symmetrized) feature-interaction matrix of the
penalized crosscoder (ckubmeg1 / λ=1000), filtered to features that have an
auto-interp explanation. For each resulting cluster (5 < size < 25) a GPT-4o judge
is shown 5 example explanations from the cluster plus 5 "test" explanations (one
held-out from the cluster, four from other clusters) and must pick the one that
belongs — 5 trials per cluster. We report accuracy for the interaction-metric (IM)
clustering vs the cosine-similarity baseline, and plot accuracy vs cluster size.
(Paper: ~73 clusters, mean accuracy ~66%.)

Inputs (precomputed artifacts): feature_interactions_ckubmeg1.npy,
activation_baseline_similarity.npy, explanations_ckubmeg1.csv.

    OPENAI_API_KEY=... python reproduce/fig3_clustering.py --data-dir /workspace/fig3data
Writes reproduce/out/fig3_clustering.pdf
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.cluster import AffinityPropagation

# Exact system prompt from autointerp_prompts.SYSTEM_INTERACTION_EVALUATOR.
JUDGE_SYS = (
    "You are a meticulous AI researcher conducting an important investigation into "
    "patterns found in language. You are analysing neurons in a language model.\n\n"
    "You will be given a list of explanations which describe the meanings of a cluster "
    "of related neurons.\n"
    "You will also be given a second list of 'test explanations', of which one belongs "
    "to the cluster of neurons.\n"
    "This list will be numbered. Your task is to determine which of the numbered "
    "explanations belongs to the cluster of neurons.\n\n"
    "You should return the number of the explanation that belongs to the cluster of "
    "neurons. Do not include any other text in your response, just a single number."
)


def cluster(matrix: np.ndarray, feature_ids: list[int]) -> dict[int, list[int]]:
    sim = (matrix + matrix.T) / 2.0
    np.fill_diagonal(sim, 1.0)
    labels = AffinityPropagation(random_state=0).fit_predict(sim)
    out: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        out.setdefault(int(lab), []).append(feature_ids[idx])
    return out


def judge_prompt(examples: list[str], tests: list[str]) -> str:
    ex = "\n".join(f"{e}\n" for e in examples)
    ts = "\n".join(f"{i}: {t}\n" for i, t in enumerate(tests))
    return f"CLUSTER EXPLANATIONS: {ex}\nTEST EXPLANATIONS: {ts}"


def accuracy_by_size(client, expl_by_cluster: dict[int, list[str]], trials: int = 5):
    """Returns list of (cluster_size, accuracy) and overall accuracy."""
    labels = list(expl_by_cluster)
    per_cluster, correct, total = [], 0, 0
    for lab, expls in expl_by_cluster.items():
        c_ok = c_tot = 0
        for _ in range(trials):
            sample = random.sample(expls, 6)
            examples, held = sample[:5], sample[5]
            others = random.sample([x for x in labels if x != lab], 4)
            tests = [random.choice(expl_by_cluster[o]) for o in others]
            pos = random.randint(0, 4)
            tests = tests[:pos] + [held] + tests[pos:]
            r = client.chat.completions.create(
                model="gpt-4o", temperature=0.7,
                messages=[{"role": "system", "content": JUDGE_SYS},
                          {"role": "user", "content": judge_prompt(examples, tests)}],
            ).choices[0].message.content.strip()
            try:
                if int(r) == pos:
                    c_ok += 1
                    correct += 1
                c_tot += 1
                total += 1
            except ValueError:
                pass
        if c_tot:
            per_cluster.append((len(expls), c_ok / c_tot))
    return per_cluster, (correct / total if total else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/fig3data")
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()
    d = Path(args.data_dir)

    im = np.load(d / "feature_interactions_ckubmeg1.npy")
    base = np.load(d / "activation_baseline_similarity.npy")
    expl = pd.read_csv(d / "explanations_ckubmeg1.csv")
    fids = [int(i) for i in expl["feature_id"].unique()]
    expl_map = {int(r.feature_id): r.explanation for r in expl.itertuples()}
    im_f = im[fids][:, fids]
    base_f = base[fids][:, fids]

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    results = {}
    for name, mat in (("IM", im_f), ("Baseline", base_f)):
        clusters = cluster(mat, fids)
        small = {k: [expl_map[f] for f in v] for k, v in clusters.items() if 5 < len(v) < 25}
        print(f"{name}: {len(clusters)} clusters total, {len(small)} of size 5-25")
        per_size, acc = accuracy_by_size(client, small, args.trials)
        print(f"{name}: mean cluster-assignment accuracy = {acc:.3f}")
        results[name] = (per_size, acc)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (per_size, acc) in results.items():
        if per_size:
            xs = [s for s, _ in per_size]
            ys = [a for _, a in per_size]
            ax.scatter(xs, ys, label=f"{name} (mean {acc:.2f})", alpha=0.7)
    ax.set_xlabel("cluster size")
    ax.set_ylabel("GPT-4o cluster-assignment accuracy")
    ax.set_title("Fig 3: interaction-metric feature clusters (ckubmeg1, λ=1000)")
    ax.legend()
    fig.tight_layout()
    out = Path(__file__).parent / "out" / "fig3_clustering.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
