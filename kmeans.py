import numpy as np
import csv
import os

outputDir = "finalClusters"

# Assuming a csv file where each line is a centroid with n dimensions
centroids = np.genfromtxt('centroids.csv', delimiter=',')
data = np.genfromtxt('data.csv', delimiter=',')

# Create a list of length k for each centroid
clusters = [[] for _ in range(centroids.shape[0])]

totalConverged = 0
maxIterations = 100
totalIterations = 0
while (totalConverged < centroids.shape[0]) and (totalIterations < maxIterations):

    # Reset the cluster arrays and reset the totalConverged for each iteration
    totalConverged = 0
    clusters = [[] for _ in range(centroids.shape[0])]

    for dp in data:
        # Calculate the norm for each row
        distances = np.linalg.norm(dp - centroids, axis=1)

        # Return one answer per row
        closestCentroid = np.argmin(distances)

        # Add to the cluster array
        clusters[closestCentroid].append(dp)


    for i in range(centroids.shape[0]):

        if len(clusters[i]) > 0:

            centroidOld = centroids[i].copy()
            centroidNew = np.mean(clusters[i], axis=0)

            converged = np.allclose(centroidOld, centroidNew, atol=1e-4)

            if converged:
                totalConverged += 1
            else:
                centroids[i] = centroidNew
        else:

            # Counting a cluster with no pints as converged
            totalConverged += 1

    if totalConverged == centroids.shape[0]:
        break
    else:
        totalIterations += 1

for i, c in enumerate(clusters):

    with open(os.path.join(outputDir, f"cluster{i}.csv"), 'w', newline='') as f:

        csvFile = csv.writer(f)
        csvFile.writerows(c)

