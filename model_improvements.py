import pickle

import optuna
from scipy.stats import spearmanr
import pacmap  # Import pacmap instead of umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

with open("person_embeddings.pkl", "rb") as f:
    person_embeddings = pickle.load(f)

scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(list(person_embeddings.values()))


# Define an objective function to be minimized.
def objective(trial):
    # Specify a search space using distributions.
    n_components = trial.suggest_categorical("n_components", [2])
    n_neighbors = trial.suggest_int("n_neighbors", 2, 15)
    mn_ratio = trial.suggest_float("MN_ratio", 0.1, 0.99)
    fp_ratio = trial.suggest_float("FP_ratio", 1.0, 5.0)

    # Create a PaCMAP instance with hyperparameters.
    reducer = pacmap.PaCMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        MN_ratio=mn_ratio,
        FP_ratio=fp_ratio
    )

    # Generate embeddings using PaCMAP
    reduced_embeddings = reducer.fit_transform(scaled_embeddings)

    # Compute cosine similarity between all pairs of students
    similarities = cosine_similarity(scaled_embeddings)

    # Compute Euclidean distance between all pairs of students in the reduced space
    distances = np.sqrt(
        np.sum(
            (reduced_embeddings[:, None] - reduced_embeddings[None, :]) ** 2, axis=-1
        )
    )

    # Rank classmates according to similarities (in descending order) and distances
    similarity_ranks = np.argsort(np.argsort(-similarities))  # Multiply by -1 to sort in descending order
    distance_ranks = np.argsort(np.argsort(distances))

    # Compute rank correlation between similarity ranks and distance ranks
    rank_correlation, _ = spearmanr(similarity_ranks, distance_ranks)

    # Return a value to be minimized (negative average rank correlation).
    avg_rank_correlation = -np.mean(rank_correlation)
    return avg_rank_correlation


# Run optimization
study = optuna.create_study()
study.optimize(objective, n_trials=200)
best_params = study.best_params

print(f"Best parameters: {best_params}")

reducer = pacmap.PaCMAP(**best_params, random_state=42)
reduced_data = reducer.fit_transform(scaled_embeddings)

# Creating lists of coordinates with accompanying labels
x = [row[0] for row in reduced_data]
y = [row[1] for row in reduced_data]
label = list(person_embeddings.keys())

# Plotting and annotating data points
plt.scatter(x, y)
for i, name in enumerate(label):
    plt.annotate(name, (x[i], y[i]), fontsize="3")

# Clean-up and Export
plt.axis("off")
plt.savefig("visualization_tuned_pacmap.png", dpi=800)
plt.show()