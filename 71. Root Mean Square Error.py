import numpy as np

def rmse(y_true, y_pred):
	error = np.sqrt(np.mean((y_true - y_pred) ** 2))
	return round(error,3)
