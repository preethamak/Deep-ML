def poly_term_derivative(c: float, x: float, n: float) -> float:
    derivative = c * n * (x ** (n - 1))
    return derivative
