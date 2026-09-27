import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method

	r = len(a)
	c = len(a[0])
	buf = [[0] * new_shape[1] for i in range(new_shape[0])]
	if r*c != new_shape[0] * new_shape[1]:
		return []
	new_row = 0
	new_col = 0
	for i in range(r):
			for j in range(c):
				if new_col == new_shape[1]:
					new_row +=1
					new_col = 0
				buf[new_row][new_col] = a[i][j]
				new_col += 1
	reshaped_matrix = buf
	return reshaped_matrix