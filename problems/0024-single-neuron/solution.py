import math
import numpy as np
def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	features = np.array(features)
	weights = np.array(weights)
	print(features.shape, weights.shape)

	res = features @ weights + bias

	def sigmod(x):
		if x>=0:
			return 1/(1+np.exp(-x))
		else:
			return np.exp(x)/(np.exp(x) + 1)
	
	for i in range(res.shape[0]):
		res[i] = sigmod(res[i])
	
	mse_loss = np.mean((res - labels) ** 2)

	return [round(each,4) for each in res.tolist()], round(mse_loss,4)

	# print(prob)

	# return probabilities, mse