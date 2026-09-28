import math
import numpy as np

def sigmoid(z: float) -> float:
	#Your code here
	# stability
	if z >= 0:
		return 1/(1+np.exp(-z))
	else:
		return np.exp(z)/(np.exp(z)+1)

	