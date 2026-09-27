def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	r = len(matrix)
	c = len(matrix[0])

	if mode == "row":
		res = [0] * r
		for i in range(r):
			res[i] = sum(matrix[i])/c
	else:
		res = [0] * c
		for i in range(c):
			s = 0
			for j in range(r):
				s += matrix[j][i]
				
			res[i] = s/r

	means = res
	return means