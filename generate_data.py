import numpy as np
import pandas as pd
import os

def generate_dgcca_cluster_data(n_samples=900, n_views=3, noise_level=0.2):
    n_clusters = 3
    samples_per_cluster = n_samples // n_clusters

    centers = np.array([[2, 0, 2], [-2, -2, 2], [0, 0, -2]])
    shared_signal = []
    labels = []

    for i in range(n_clusters):
        cluster_data = centers[i] + np.random.randn(samples_per_cluster, 3) * 0.6
        shared_signal.append(cluster_data)
        labels.append(np.full(samples_per_cluster, i))

    shared_signal = np.concatenate(shared_signal, axis=0)
    labels = np.concatenate(labels, axis=0)

    n_samples_actual = shared_signal.shape[0]
    idx = np.random.permutation(n_samples_actual)
    shared_signal, labels = shared_signal[idx], labels[idx]

    views = []
    for v in range(n_views):
        proj = np.random.randn(3, 10)
        view_data = np.dot(shared_signal, proj)

        if v == 0:
            view_data = np.exp(view_data / 10.0)
        elif v == 1:
            view_data = np.sin(view_data * 2)
        else:
            view_data = np.abs(view_data)**1.5

        view_data += np.random.randn(n_samples_actual, 10) * noise_level
        views.append(view_data)

    return views, labels

def save_synthetic_data(output_dir="./data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    views, labels = generate_dgcca_cluster_data(n_samples=900, n_views=3)

    for i, view_data in enumerate(views):
        filename = os.path.join(output_dir, f"view{i+1}.csv")
        pd.DataFrame(view_data).to_csv(filename, index=False, header=False)
        print(f"Saved: {filename}")

    # Optional: Save labels for validation later
    pd.DataFrame(labels).to_csv(os.path.join(output_dir, "labels.csv"), index=False, header=False)

if __name__ == "__main__":
    save_synthetic_data()