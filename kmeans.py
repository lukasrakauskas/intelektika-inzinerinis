import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import math

df = pd.read_csv('heart.csv')
# pd.plotting.scatter_matrix(df)
# plt.show()

column = "age"
b = "thalachh"
rows = ["trtbps", "chol", "thalachh", "oldpeak"]

rows_inertia = {
    "trtbps": [],
    "chol": [],
    "thalachh": [],
    "oldpeak": [],
}
# K = 3

# Select random observation as centroids


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
        Centroids_new = X.groupby(["Cluster"]).mean()[
            [b, column]]
        if j == 0:
            diff = 1
            j = j+1
        else:
            diff = (Centroids_new[b] - Centroids[b]).sum() + \
                (Centroids_new[column] -
                 Centroids[column]).sum()
            print(diff.sum())
        Centroids = X.groupby(["Cluster"]).mean()[
            [b, column]]

    color = ['blue', 'green', 'cyan', 'purple', 'orange']
    
    for k in range(K):
        data = X[X["Cluster"] == k+1]

        centroid = Centroids.iloc[k]
        x2 = centroid[column]
        y2 = centroid[b]
        for i, row_data in data.iterrows():
            x1 = row_data[column]
            y1 = row_data[b]
            total_inertia += euclidean(x1, y1, x2, y2)
        
        plt.scatter(data[column], data[b], c=color[k])

    rows_inertia[b].append(total_inertia)

    plt.scatter(Centroids[column], Centroids[b], c='red')
    plt.xlabel('age')
    plt.ylabel(b)
    plt.show()


def euclidean(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


centroids(3, "chol")


for K in range(3, 6):
    for row in rows:
        centroids(K, row)

print(rows_inertia)