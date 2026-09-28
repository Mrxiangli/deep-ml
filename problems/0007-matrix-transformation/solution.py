import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	A=np.array(A)
	T = np.array(T)
	S = np.array(S)
	def is_invertable(m):
		if m.shape[0]!=m.shape[1]:
			return False
		return np.linalg.matrix_rank(m) == m.shape[0]
	
	if is_invertable(T) and is_invertable(S):
		return np.linalg.inv(T)@ A @ S
	return -1
