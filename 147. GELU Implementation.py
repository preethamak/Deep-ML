import numpy as np

def GeLU(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * np.power(x, 3))))
    '''GELU is not a built in function in numpy, so write the formula manually'''
