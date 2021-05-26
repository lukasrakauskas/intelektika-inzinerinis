import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

df = pd.read_csv('heart.csv')
# pd.plotting.scatter_matrix(df)
# plt.show()

column = "age"
b = "thalachh"
rows = ["trtbps", "chol", "thalachh", "oldpeak"]


def centroids(K, b):
    X = df[[column, b]]
    Centroids = (X.sample(n=K))

    total_inertia = 0

    diff = 1
    j = 0

    while(diff != 0):
        XD = X
        i = 1
        for index1, row_c in Centroids.iterrows():
            ED = []
            for index2, row_d in XD.iterrows():
                d1 = (row_c[column]-row_d[column])**2
                d2 = (row_c[b]-row_d[b])**2
                d = np.sqrt(d1+d2)
                ED.append(d)
            X[i] = ED
            i = i+1

        C = []
        for index, row in X.iterrows():
            min_dist = row[1]
            pos = 1
            for i in range(K):
                if row[i+1] < min_dist:
                    min_dist = row[i+1]
                    pos = i+1
            C.append(pos)
        X["Cluster"] = C
        # X.loc[:, "Cluster"] = C
        Centroids_new = X.groupby(["Cluster"]).mean()[
            [b, column]]
        if j == 0:
            diff = 1
            j = j+1
        else:
            diff = (Centroids_new[b] - Centroids[b]).sum() + \
                (Centroids_new[column] -
                 Centroids[column]).sum()
            # print(diff.sum())
        Centroids = X.groupby(["Cluster"]).mean()[
            [b, column]]

    color = ['blue', 'green', 'cyan', 'purple', 'orange']

    for k in range(K):
        data = X[X["Cluster"] == k + 1]
        other_clusters = X[X["Cluster"] != k + 1]

        silhouette_coeffs = []

        for i, sample in data.iterrows():
            ai = calculate_ai(sample, data, column, b)
            bi = calculate_bi(sample, other_clusters, column, b)
            Si = (bi - ai) / max(bi, ai)
            silhouette_coeffs.append(Si)

        silhouette[b].append(silhouette_coeffs)

        centroid = Centroids.iloc[k]
        total_inertia += calculate_inertia(centroid, data, column, b)

        # plt.scatter(data[column], data[b], c=color[k])

    rows_inertia[b].append(total_inertia)

    # plt.scatter(Centroids[column], Centroids[b], c='red')
    # plt.xlabel('age')
    # plt.ylabel(b)
    # plt.show()

    return X


def calculate_ai(sample, data, column, row):
    x1 = sample[column]
    y1 = sample[row]

    ai = 0

    for i, row_data in data.iterrows():
        x2 = row_data[column]
        y2 = row_data[row]
        ai += euclidean(x1, y1, x2, y2)

    return ai / len(data)

def calculate_bi(sample, other_clusters, column, row):
    x1 = sample[column]
    y1 = sample[row]

    clusters = other_clusters.Cluster.unique()

    mean_distances = []

    for cluster in clusters:
        data = other_clusters[other_clusters.Cluster == cluster]
        mean_distance = 0
        for i, row_data in data.iterrows():
            x2 = row_data[column]
            y2 = row_data[row]
            mean_distance += euclidean(x1, y1, x2, y2)
        mean_distances.append(mean_distance / len(data))

    return min(mean_distances)


def calculate_inertia(centroid, data, column, row):
    total_inertia = 0

    x2 = centroid[column]
    y2 = centroid[row]
    for i, row_data in data.iterrows():
        x1 = row_data[column]
        y1 = row_data[row]
        total_inertia += euclidean(x1, y1, x2, y2)

    return total_inertia


def euclidean(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def plot_silhouette_graphs(K, silhouette_values, data, column, row):
    X = 0
    for i in silhouette_values:
        X += len(i)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, X + (K + 1) * 10])

    y_lower = 10
    for i in range(K):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = silhouette_values[i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = len(ith_cluster_silhouette_values)
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / K)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=(sum(ith_cluster_silhouette_values) / len(ith_cluster_silhouette_values)), color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    # colors = cm.nipy_spectral(float(i) / K)
    for i in range(K):
        current_data = data[data["Cluster"] == i+1]
        ax2.scatter(current_data[column].to_numpy(), current_data[row].to_numpy(), marker='.', s=30, lw=0, alpha=0.7,
                 edgecolor='k')

        centerX = np.average(current_data[column]) 
        centerY = np.average(current_data[row]) 
        ax2.scatter(centerX, centerY, marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
        ax2.scatter(centerX, centerY, marker='$%d$' % (i+1), alpha=1,
                    s=50, edgecolor='k')

    # Labeling the clusters
    # centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers

        

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % K),
                 fontsize=14, fontweight='bold')

    plt.show()


# centroids(3, 'chol')
# plot_silhouette_graphs(3, len(silhouette), silhouette)

rows_inertia = {
    "trtbps": [],
    "chol": [],
    "thalachh": [],
    "oldpeak": [],
}

K = 3
silhouette = {
    "trtbps": [],
    "chol": [],
    "thalachh": [],
    "oldpeak": [],
}

X = centroids(K, 'chol')
plot_silhouette_graphs(K, silhouette['chol'], X, 'age', 'chol')



# for K in range(3, 6):
    # for row in rows:


# for key, value in rows_inertia.items():
#     plt.plot(x, value)
#     plt.show()