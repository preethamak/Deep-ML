def determinant_4x4(matrix: list[list[int | float]]) -> float:
	if len(matrix) == 2:
		return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

	det = 0
	for j in range(len(matrix)):
		minor = [row[:j] + row[j+1:] for row in matrix[1:]]
		det += ((-1)**j) * matrix[0][j] * determinant_4x4(minor)

	return det
