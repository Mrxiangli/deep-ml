def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	r = len(matrix)
	c = len(matrix[0])

	for i in range(r):
		for j in range(c):
			matrix[i][j]*= scalar
	return matrix