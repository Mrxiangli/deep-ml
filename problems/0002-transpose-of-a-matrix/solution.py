def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    r = len(a)
    c = len(a[0])

    new = [[0] * r for i in range(c)]

    for i in range(r):
        for j in range(c):
            new[j][i] = a[i][j]
    
    return new