import numpy as np

# Assuming a csv file where each line is a centroid with n dimensions

centroids = np.genfromtxt('centroids.csv', delimiter=',')
data = np.genfromtxt('data.csv', delimiter=',')

# Create a list of length k for each centroid
clusters = [[] for _ in range(centroids.shape[0])]

totalConverged = 0
while totalConverged < centroids.shape[0]:
    for dp in data:
        # Calculate the norm for each row
        distances = np.linalg.norm(dp - centroids, axis=1)

        # Return one answer per row
        closestCentroid = np.argmin(distances)

        # Add to the cluster array
        clusters[closestCentroid].append(dp)


    for i in range(centroids.shape[0]):

        if len(clusters[i]) > 0:

            centroidOld = np.array(centroids[i])
            centroidNew = np.array(np.mean(clusters[i], axis=0))

            converged = np.allclose(centroidOld, centroidNew, atol=1e-4)

            if converged:
                totalConverged += 1
            else:
                centroids[i] = centroidNew
        else:
            totalConverged += 1

