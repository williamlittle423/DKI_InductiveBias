import os
import numpy as np
import pandas as pd


DATA_DIR = "data"
RESULTS_DIR = "results"

PTRAIN_PATH = os.path.join(DATA_DIR, "Ptrain.csv")
SAMPLE_ID_PATH = os.path.join(DATA_DIR, "Sample_id.csv")
SPECIES_ID_PATH = os.path.join(DATA_DIR, "Species_id.csv")


def bray_curtis(p, q, eps=1e-8):
    return np.sum(np.abs(p - q)) / (np.sum(np.abs(p + q)) + eps)


def load_id_vector(path):
    ids = np.loadtxt(path, delimiter=",").astype(int)

    if ids.ndim > 1:
        ids = ids.squeeze()

    return ids


def ensure_species_by_samples(Q, n_species):
    """
    Expected DKI format:
        Q shape = species x perturbation_samples

    But your training script may save:
        Q shape = perturbation_samples x species

    This fixes that automatically.
    """
    if Q.shape[0] == n_species:
        return Q

    if Q.shape[1] == n_species:
        return Q.T

    raise ValueError(
        f"Could not infer orientation for Q with shape {Q.shape} "
        f"and n_species={n_species}"
    )


def compute_model_keystoneness(
    model_name,
    P_original,
    Q_perturbed,
    sample_ids,
    species_ids,
    ids_are_one_indexed=True,
):
    """
    Returns dataframe with:
        sample_id, species_id, keystoneness, model
    """

    n_species = P_original.shape[0]
    Q_perturbed = ensure_species_by_samples(Q_perturbed, n_species)

    if ids_are_one_indexed:
        sample_lookup = sample_ids - 1
        species_lookup = species_ids - 1
    else:
        sample_lookup = sample_ids
        species_lookup = species_ids

    rows = []

    for k in range(Q_perturbed.shape[1]):
        sample_idx = sample_lookup[k]
        species_idx = species_lookup[k]

        p_original = P_original[:, sample_idx]
        q_removed = Q_perturbed[:, k]

        ks = bray_curtis(p_original, q_removed)

        rows.append(
            {
                "sample_id": int(sample_ids[k]),
                "species_id": int(species_ids[k]),
                "keystoneness": float(ks),
                "model": model_name,
            }
        )

    return pd.DataFrame(rows)


def compute_all_model_keystoneness(model_names):
    P_original = np.loadtxt(PTRAIN_PATH, delimiter=",")
    P_original = P_original / (P_original.sum(axis=0, keepdims=True) + 1e-8)

    sample_ids = load_id_vector(SAMPLE_ID_PATH)
    species_ids = load_id_vector(SPECIES_ID_PATH)

    all_results = []

    for model_name in model_names:
        q_path = os.path.join(RESULTS_DIR, f"qtst_{model_name}.csv")

        if not os.path.exists(q_path):
            print(f"Skipping {model_name}: missing {q_path}")
            continue

        Q_perturbed = np.loadtxt(q_path, delimiter=",")

        ks_df = compute_model_keystoneness(
            model_name=model_name,
            P_original=P_original,
            Q_perturbed=Q_perturbed,
            sample_ids=sample_ids,
            species_ids=species_ids,
            ids_are_one_indexed=True,
        )

        all_results.append(ks_df)

    if len(all_results) == 0:
        raise RuntimeError("No model keystoneness files were computed.")

    return pd.concat(all_results, ignore_index=True)


model_names = [
    "cnode",
    "node",
    "linear",
    "residualmlp",
    "attention",
]

keystoneness_df = compute_all_model_keystoneness(model_names)

output_path = os.path.join(RESULTS_DIR, "keystoneness_by_model.csv")
keystoneness_df.to_csv(output_path, index=False)

print(keystoneness_df.head())
print(f"Saved: {output_path}")