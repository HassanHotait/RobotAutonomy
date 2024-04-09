import numpy as np

def compute_rotation_matrix(pc1, pc2):
    # Calculate covariance matrix of pointcloud
    cov = np.cov(pc1 @ pc2.T)
    # Perform SVD
    U, S, Vt = np.linalg.svd(cov)
    # Calculate rotation matrix R
    R = (U @ Vt).reshape(2, 2)
    return R

def compute_rotation_matrix_regularized(pc1, pc2, epsilon=1e-8):
    # Calculate covariance matrix of pointcloud
    cov = np.cov(pc1 @ pc2.T)
    # Add a small positive constant to the diagonal for regularization
    cov += np.eye(cov.shape[0]) * epsilon
    # Perform SVD
    U, S, Vt = np.linalg.svd(cov)
    # Calculate rotation matrix R
    R = (U @ Vt).reshape(2, 2)
    return R

# Example usage
pc1 = np.random.rand(2, 10).astype(np.float64)
pc2 = pc1

print('Point Clouds')
print(f'Rotation Matrix: \n {compute_rotation_matrix(pc1, pc2)}')

print('Point Clouds with Regularization')
print(f'Rotation Matrix: \n {compute_rotation_matrix_regularized(pc1, pc2)}')