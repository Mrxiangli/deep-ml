import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	res = np.array(vectors)
	cv = np.cov(res)
	return cv.tolist()