import pandas as pd
import numpy as np
from scipy.stats import spearmanr


RESULTS_PATH = "results/keystoneness_by_model.csv"


def topk_jaccard(rank_a, rank_b, k=10):

    top_a = set(rank_a.head(k).index)
    top_b = set(rank_b.head(k).index)

    return len(top_a & top_b) / len(top_a | top_b)


# -----------------------------
# Load keystoneness data
# -----------------------------
df = pd.read_csv(RESULTS_PATH)

print(df.head())


# -----------------------------
# Compute global median rankings
# -----------------------------
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

        print("\n")
        print("=" * 60)
        print(f"{model_a} vs {model_b}")
        print("=" * 60)

        print(f"Spearman rho: {rho:.4f}")
        print(f"p-value: {pval:.4e}")
        print(f"Top-10 Jaccard: {jaccard_10:.4f}")
        print(f"Top-20 Jaccard: {jaccard_20:.4f}")


# -----------------------------
# Save results
# -----------------------------
comparison_df = pd.DataFrame(comparison_rows)

comparison_df.to_csv(
    "results/model_ranking_comparisons.csv",
    index=False,
)

print("\nSaved:")
print("results/model_ranking_comparisons.csv")

import matplotlib.pyplot as plt


# -----------------------------
# Build comparison matrices
# -----------------------------
spearman_matrix = pd.DataFrame(
    np.eye(len(models)),
    index=models,
    columns=models,
)

jaccard10_matrix = pd.DataFrame(
    np.eye(len(models)),
    index=models,
    columns=models,
)

for row in comparison_rows:
    a = row["model_a"]
    b = row["model_b"]

    spearman_matrix.loc[a, b] = row["spearman_rho"]
    spearman_matrix.loc[b, a] = row["spearman_rho"]

    jaccard10_matrix.loc[a, b] = row["top10_jaccard"]
    jaccard10_matrix.loc[b, a] = row["top10_jaccard"]


# -----------------------------
# Plot helper
# -----------------------------
import matplotlib.pyplot as plt

# Increase global font sizes
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# -----------------------------
# Plot helper
# -----------------------------
def plot_matrix(matrix, title, output_path, vmin=0, vmax=1):
    fig, ax = plt.subplots(figsize=(9, 8))

    im = ax.imshow(
        matrix.values,
        vmin=vmin,
        vmax=vmax,
        cmap="managua",
    )

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))

    ax.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right",
        fontsize=16,
    )

    ax.set_yticklabels(
        matrix.index,
        fontsize=16,
    )

    # Cell text
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix.iloc[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

    # Grid aligned to cell borders
    ax.set_xticks(
        np.arange(-0.5, len(matrix.columns), 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, len(matrix.index), 1),
        minor=True
    )

    ax.grid(
        which="minor",
        color="black",
        linestyle="-",
        linewidth=1
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    ax.set_title(
        title,
        fontsize=20,
        pad=20,
    )

    # Bigger colorbar labels
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


# -----------------------------
# Make plots
# -----------------------------
plot_matrix(
    spearman_matrix,
    "Spearman Correlation of Global Keystone Rankings",
    "results/spearman_correlation_matrix.png",
    vmin=0,
    vmax=1,
)

plot_matrix(
    jaccard10_matrix,
    "Top-10 Jaccard Similarity of Keystone Species",
    "results/top10_jaccard_matrix.png",
    vmin=0,
    vmax=1,
)

print("Saved:")
print("results/spearman_correlation_matrix.png")
print("results/top10_jaccard_matrix.png")