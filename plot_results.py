import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd
import numpy as np

os.makedirs("results/figures", exist_ok=True)
RESULTS_PATH = "results/keystoneness_by_model.csv"

# Increase global font sizes
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

def topk_jaccard(rank_a, rank_b, k=10):

    top_a = set(rank_a.head(k).index)
    top_b = set(rank_b.head(k).index)

    return len(top_a & top_b) / len(top_a | top_b)

# -----------------------------
# Load keystoneness data
# -----------------------------
df = pd.read_csv(RESULTS_PATH)

global_rankings = {}

for model_name in df["model"].unique():

    model_df = df[df["model"] == model_name]

    ranking = (
        model_df
        .groupby("species_id")["keystoneness"]
        .median()
        .sort_values(ascending=False)
    )

    global_rankings[model_name] = ranking

    print("\n")
    print("=" * 60)
    print(f"TOP KEYSTONE SPECIES: {model_name}")
    print("=" * 60)
    print(ranking.head(10))


# -----------------------------
# Compare models
# -----------------------------
models = list(global_rankings.keys())

comparison_rows = []

for i in range(len(models)):
    for j in range(i + 1, len(models)):

        model_a = models[i]
        model_b = models[j]

        rank_a = global_rankings[model_a]
        rank_b = global_rankings[model_b]

        common_species = rank_a.index.intersection(rank_b.index)

        rho, pval = spearmanr(
            rank_a.loc[common_species],
            rank_b.loc[common_species],
        )

        jaccard_10 = topk_jaccard(rank_a, rank_b, k=10)
        jaccard_20 = topk_jaccard(rank_a, rank_b, k=20)

        comparison_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "spearman_rho": rho,
                "spearman_pval": pval,
                "top10_jaccard": jaccard_10,
                "top20_jaccard": jaccard_20,
            }
        )



# ============================================================
# 3. Scatter plot: cNODE vs another model
# ============================================================
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd
import numpy as np

os.makedirs("results/figures", exist_ok=True)
RESULTS_PATH = "results/keystoneness_by_model.csv"

def topk_jaccard(rank_a, rank_b, k=10):

    top_a = set(rank_a.head(k).index)
    top_b = set(rank_b.head(k).index)

    return len(top_a & top_b) / len(top_a | top_b)

# -----------------------------
# Load keystoneness data
# -----------------------------
df = pd.read_csv(RESULTS_PATH)

global_rankings = {}

for model_name in df["model"].unique():

    model_df = df[df["model"] == model_name]

    ranking = (
        model_df
        .groupby("species_id")["keystoneness"]
        .median()
        .sort_values(ascending=False)
    )

    global_rankings[model_name] = ranking

    print("\n")
    print("=" * 60)
    print(f"TOP KEYSTONE SPECIES: {model_name}")
    print("=" * 60)
    print(ranking.head(10))


# -----------------------------
# Compare models
# -----------------------------
models = list(global_rankings.keys())

comparison_rows = []

for i in range(len(models)):
    for j in range(i + 1, len(models)):

        model_a = models[i]
        model_b = models[j]

        rank_a = global_rankings[model_a]
        rank_b = global_rankings[model_b]

        common_species = rank_a.index.intersection(rank_b.index)

        rho, pval = spearmanr(
            rank_a.loc[common_species],
            rank_b.loc[common_species],
        )

        jaccard_10 = topk_jaccard(rank_a, rank_b, k=10)
        jaccard_20 = topk_jaccard(rank_a, rank_b, k=20)

        comparison_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "spearman_rho": rho,
                "spearman_pval": pval,
                "top10_jaccard": jaccard_10,
                "top20_jaccard": jaccard_20,
            }
        )



# ============================================================
# 3. Scatter plot: cNODE vs another model
# ============================================================
def plot_keystoneness_scatter(global_rankings, model_x, model_y):
    rank_x = global_rankings[model_x]
    rank_y = global_rankings[model_y]

    common_species = rank_x.index.intersection(rank_y.index)

    x = rank_x.loc[common_species]
    y = rank_y.loc[common_species]

    rho, pval = spearmanr(x, y)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.75)

    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(f"{model_x} median keystoneness")
    plt.ylabel(f"{model_y} median keystoneness")
    plt.title(
        f"{model_x} vs {model_y}\n"
        f"Spearman ρ = {rho:.3f}, p = {pval:.1e}"
    )

    plt.tight_layout()

    output_path = f"results/figures/scatter_{model_x}_vs_{model_y}.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved: {output_path}")


plot_keystoneness_scatter(global_rankings, "cnode", "linear")
plot_keystoneness_scatter(global_rankings, "cnode", "attention")
plot_keystoneness_scatter(global_rankings, "cnode", "residualmlp")


# ============================================================
# 4. Bar plot of top global keystone species
# ============================================================
# ============================================================
# Multi-panel scatter plots
# ============================================================
def plot_all_keystoneness_scatters(
    global_rankings,
    reference_model="cnode",
    comparison_models=None,
):

    if comparison_models is None:
        comparison_models = [
            m for m in global_rankings.keys()
            if m != reference_model
        ]

    n_models = len(comparison_models)

    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(6 * n_models, 6),
    )

    if n_models == 1:
        axes = [axes]

    for ax, model_y in zip(axes, comparison_models):

        rank_x = global_rankings[reference_model]
        rank_y = global_rankings[model_y]

        common_species = rank_x.index.intersection(rank_y.index)

        x = rank_x.loc[common_species]
        y = rank_y.loc[common_species]

        rho, pval = spearmanr(x, y)

        ax.scatter(x, y, alpha=0.75)

        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())

        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            linewidth=1,
            color="black",
        )

        ax.set_xlabel(f"{reference_model} keystoneness")
        ax.set_ylabel(f"{model_y} keystoneness")

        ax.set_title(
            f"{reference_model} vs {model_y}\n"
            f"ρ = {rho:.3f}"
        )

    plt.tight_layout()

    output_path = (
        f"results/figures/"
        f"scatter_all_{reference_model}_comparisons.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved: {output_path}")


plot_all_keystoneness_scatters(
    global_rankings,
    reference_model="cnode",
    comparison_models=[
        "linear",
        "attention",
        "residualmlp",
    ]
)


# ============================================================
# 4. Bar plot of top global keystone species
# ============================================================
def plot_top_keystone_species(global_rankings, reference_model="cnode", top_k=10):
    top_species = (
        global_rankings[reference_model]
        .head(top_k)
        .index
        .tolist()
    )

    plot_df = []

    for model_name, ranking in global_rankings.items():
        for species_id in top_species:
            plot_df.append(
                {
                    "model": model_name,
                    "species_id": species_id,
                    "keystoneness": ranking.loc[species_id],
                }
            )

    plot_df = pd.DataFrame(plot_df)

    species_labels = [str(s) for s in top_species]
    models = list(global_rankings.keys())

    x = np.arange(len(top_species))
    width = 0.8 / len(models)

    plt.figure(figsize=(12, 6))

    for i, model_name in enumerate(models):
        model_vals = (
            plot_df[plot_df["model"] == model_name]
            .set_index("species_id")
            .loc[top_species]["keystoneness"]
            .values
        )

        plt.bar(
            x + i * width,
            model_vals,
            width,
            label=model_name,
        )

    plt.xticks(
        x + width * (len(models) - 1) / 2,
        species_labels,
        rotation=45,
        ha="right",
    )

    plt.xlabel("Species ID")
    plt.ylabel("Median keystoneness")
    plt.title(f"Top {top_k} Global Keystone Species by {reference_model}")
    plt.legend()

    plt.tight_layout()

    output_path = f"results/figures/top{top_k}_keystone_species_by_{reference_model}.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved: {output_path}")


plot_top_keystone_species(
    global_rankings,
    reference_model="cnode",
    top_k=10,
)