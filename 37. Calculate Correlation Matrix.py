import numpy as np

def calculate_correlation_matrix(X, Y=None):
	# Your code here
	X = np.asarray(X, dtype= float)
    n_samp = X.shape[0]

    X_cen = X - X.mean(axis=0, keepdims= True)

    X_std = X_cen.std(axis=0, ddof=0, keepdims= True)

    if Y is None:
        Y_cen = X_cen
        Y_std = X_std

    else:
        Y = np.asarray(Y, dtype= float)
        Y_cen = Y - Y.mean(axis= 0, keepdims= True)
        Y_std = Y_cen.std(axis= 0, ddof= 0, keepdims= True)

    cov = (X_cen.T @ Y_cen)/n_samp
    corr_matrix = cov/(X_std.T @ Y_std)

    return corr_matrix
