import numpy as np
import random


class NK(object):
    def __init__(self, k=2, epochs=100, error=0.1):
        self.k = k
        self.epochs = epochs
        self.error = error
        self.pre_error = 0
        self.centroids = []
        self.iteration = 0
        self.sse = 0.0

    def distance(self, centroid, point):
        n = len(centroid)
        return np.sum((centroid - point) ** n) ** (1 / n)

    def squared_distance(self, centroid, point):
        return np.sqrt(np.sum((centroid - point) ** 2))

    def find_centroids(self, data):
        if len(self.centroids) == 0:
            self.centroids.append(data[random.randint(0, len(data))])
        max_dist_point = None
        max_dist = 0
        for d in data:
            for centroid in self.centroids:
                dis = self.squared_distance(centroid, d)
                if dis > max_dist:
                    max_dist = dis
                    max_dist_point = d
        self.centroids.append(max_dist_point)
        if len(self.centroids) < self.k:
            self.find_centroids(data)

    def fit(self, data):
        clusters = {}
        self.centroids = []

        # Get the centroids according to kmeans++ algorithm
        self.find_centroids(data)

        for epoch in range(self.epochs):
            self.iteration += 1
            for i in range(self.k):
                clusters[i] = [[0] * data.shape[1]]

            # Get the nearest data points to the centroid
            for d in data:
                distances = [self.squared_distance(centroid, d) for centroid in self.centroids]
                cluster_index = distances.index(min(distances))
                clusters[cluster_index].append(d)
            # update the centroids
            previous_centroid = self.centroids.copy()
            for i in range(self.k):
                self.centroids[i] = np.array(clusters[i]).mean(axis=0)
            error = np.sum(np.square(np.array(previous_centroid) - np.array(self.centroids)))
            print("Epoch " + str(epoch) + " Loss = " + str(abs(error)))
            if self.iteration == 1:
                self.sse = error / self.k
            if abs(error) < self.error or abs(self.pre_error - error) < 1e-6:
                break
            self.pre_error = error

    def predict(self, data):
        preds = []
        for d in data:
            distances = [self.distance(centroid, d) for centroid in self.centroids]
            cluster_index = distances.index(min(distances))
            preds.append(cluster_index)
        return np.array(preds)


###

# Python implementaion of A Linear Time-Complexity k-Means Algorithm Using Cluster Shifting with Kmean++ initialization

###


class LK(object):
    def __init__(self, k=2, epochs=100, error=0.01, alpha=0.01):
        self.k = k
        self.epochs = epochs
        self.error = error
        self.centroids = []
        self.alpha = alpha
        self.sum_of_list_of_update_vectors = [0] * k
        self.pre_error = 0
        self.iteration = 0
        self.sse = 0.0

    def distance(self, centroid, point):
        n = len(centroid)
        return (np.sum(centroid - point) ** n) ** (1 / n)

    def squared_distance(self, centroid, point):
        return np.sqrt(np.sum((centroid - point) ** 2))

    def find_centroids(self, data):
        if len(self.centroids) == 0:
            self.centroids.append(data[random.randint(0, len(data))])
        max_dist_point = None
        max_dist = 0
        for d in data:
            for centroid in self.centroids:
                dis = self.squared_distance(centroid, d)
                if dis > max_dist:
                    max_dist = dis
                    max_dist_point = d
        self.centroids.append(max_dist_point)
        if len(self.centroids) < self.k:
            self.find_centroids(data)

    def fit(self, data):
        clusters = {}
        shape = data.shape
        self.centroids = []

        # Get the centroids according to kmeans++ algorithm
        self.find_centroids(data)

        for i in range(self.k):
            self.sum_of_list_of_update_vectors[i] = 0
        for epoch in range(self.epochs):
            self.iteration += 1
            # Resetting Clusters
            for i in range(self.k):
                clusters[i] = [[0] * data.shape[1]]

            # Get the nearest data points to the centroid
            for d in data:
                distances = [self.squared_distance(centroid, d) for centroid in self.centroids]
                cluster_index = distances.index(min(distances))
                clusters[cluster_index].append(d)

            global_centroid = np.array([np.mean(data[:, i]) for i in range(data.shape[1])])

            # Calculating direction vectors and update vector

            direction_vector = self.centroids - global_centroid
            update_vector = self.alpha * direction_vector
            temp_arr = np.array([])
            for i in range(self.k):
                self.sum_of_list_of_update_vectors[i] += update_vector[i]
                updated_data_points = np.array(clusters[i]) + update_vector[i]
                self.centroids[i] += update_vector[i]
                temp_arr = np.append(temp_arr, updated_data_points)
            # data = temp_arr.reshape(shape)

            # update the centroids
            previous_centroid = self.centroids.copy()
            for i in range(self.k):
                self.centroids[i] = np.array(clusters[i]).mean(axis=0)

            error = np.sum(np.array(previous_centroid) - np.array(self.centroids))
            print("Epoch " + str(epoch) + " Loss = " + str(abs(error)))
            self.sse = error
            if abs(error) < self.error or abs(self.pre_error - error) < 1e-4:
                break
            self.pre_error = error

    # Returning the cluster classes for each data
    def predict(self, data):
        prediction = []
        for d in data:
            distances = [self.distance(self.centroids[index], d + self.sum_of_list_of_update_vectors[index]) for index
                         in range(len(self.centroids))]
            prediction.append(distances.index(min(distances)))
        return np.array(prediction)

# from Linear_kmeans import KMeans
# import Normal_kmeans as nk


# df = pd.read_csv('wine.csv',header = None)
# header=['class','Alcohol',' Malic acid', 'Ash', 'Alcalinity', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines','Proline']
# df.columns = header
# X = df[df.columns[df.columns!='class']].values
# y = df['class'].values
# scaler = StandardScaler()
# X_new = scaler.fit_transform(X)
# X_train,X_test,y_train,y_test = train_test_split(X_new,y,test_size = 0.2, random_state = 12)
#
# print("Training Linear Kmeans")
# start_time = time.time()
# kmeans = LK(k=3)
# kmeans.fit(X_train)
# print("Time Taken for Linear Kmeans:", time.time() - start_time)
#
# print("Training Normal Kmeans")
# start_time = time.time()
# n_kmeans = NK(k=3)
# n_kmeans.fit(X_train)
# print("Time Taken by Normal Kmeans:", time.time() - start_time)
