import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    A = np.array(A, dtype=float)
    ATA = A.T @ A
    a, b, c = ATA[0, 0], ATA[0, 1], ATA[1, 1]

    # Jacobi rotation
    phi = 0.5 * np.arctan2(2 * b, a - c)
    cos = np.cos(phi)
    sin = np.sin(phi)
    R = np.array([[cos, -sin],
                  [sin,  cos]])

    # Diagonalize
    D = R.T @ ATA @ R
    S = np.sqrt(np.diag(D))

    # Sort descending
    order = np.argsort(S)[::-1]
    S = S[order]
    V = R[:, order]

    # Compute U
    U = np.zeros((2, 2))
    for i in range(2):
        if S[i] > 1e-12:
            U[:, i] = (A @ V[:, i]) / S[i]
        else:
            # pick orthonormal vector
            if i == 0:
                U[:, i] = np.array([1, 0])
            else:
                U[:, i] = np.array([-U[1, 0], U[0, 0]])
        # normalize
        U[:, i] /= np.linalg.norm(U[:, i])

    # --- Deterministic sign fix per column ---
    for i in range(2):
        if np.dot(U[:, i], A @ V[:, i]) < 0:
            U[:, i] *= -1
            V[:, i] *= -1

    return U, S, V.T
