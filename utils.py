import numpy as np

from tqdm import tqdm


EPOCHS = 5 #number of epochs
ITERATIONS = 15 #number of iterations per epoch
DEBUG = True #makes some logs if True

def kmeans(data, k):
	"""KMeans algorithm implementation in third dimension
	
	Args:
		data (np.ndarray): data of an image
		k (int): number of clusters to find

	Returns:
		numpy.ndarray: clusters
	"""
	def iterate(data, centroids):
		"""KMeans iteration
		
		1. assign points to centroids
		2. move centroids to their new position (mean of assigned points)
		3. calculate sum of distances between points and their centroids (loss value)

		Args:
			data (np.ndarray): data of an image
			centroids (np.ndarray): centroids positions
		
		Returns:
			tuple (clusters(numpy.ndarray), loss value(float), centroids(np.ndarray))
		"""
		distances = np.zeros((len(data), len(centroids)))
		
		for i in range(k):
			distances[:, i] = np.linalg.norm(data - centroids[i], axis=1)

		clusters = np.argmin(distances, axis=1)
		labels = np.zeros(len(centroids), dtype=int)

		for i in range(k):
			mask = (clusters == i)
			centroids[i] = np.mean(data[mask], axis=0)
			labels[i] += sum(mask)

		return labels, centroids

	assert isinstance(data, np.ndarray), 'Data must be instance of numpy'
	assert isinstance(k, int), 'Number of clusters must be an int'
	assert data.dtype == np.uint8, 'Data must be in uint8 format'

	data = data.reshape(-1, 3)
	centroids = np.random.uniform(0, 256, (k, 3)) #randomly generates centroids

	labels, centroids = iterate(data, centroids)

	print(centroids[:5])

	return labels, centroids


def euclidian_dist(p1, p2):
		"""Find euclidian distance between two points (p1 and p2)
		
		Args:
			p1 (np.ndarray): first point
			p2 (np.ndarray): second point

		Returns:
			float: euclidian distance
		"""
		#distance = sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

		dist = np.subtract(p1, p2)
		dist = np.power(dist, 2)
		dist = np.sum(dist)
		dist = np.sqrt(dist)

		return dist


