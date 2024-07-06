import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matrix_A = np.array([[1, 2], [3, 4], [5, 6]])


def svd_realization(matrix_A):
    at_a = np.dot(matrix_A.T, matrix_A)
    eigen_values_v, eigen_vectors_v = np.linalg.eigh(at_a)

    sorted_ind = np.argsort(eigen_values_v)[::-1]
    eigen_values_v = eigen_values_v[sorted_ind]
    eigen_vectors_v = eigen_vectors_v[:, sorted_ind]
    V = eigen_vectors_v

    sigma_values = np.sqrt(eigen_values_v)

    a_at = np.dot(matrix_A, matrix_A.T)
    eigen_values_u, eigen_vectors_u = np.linalg.eigh(a_at)

    sorted_ind = np.argsort(eigen_values_u)[::-1]
    eigen_values_u = eigen_values_u[sorted_ind]
    eigen_vectors_u = eigen_vectors_u[:, sorted_ind]
    U = eigen_vectors_u

    E = np.zeros_like(matrix_A, dtype=float)
    np.fill_diagonal(E, sigma_values)

    return U, E, V.T


U, E, V_T = svd_realization(matrix_A)
print(f"Matrix A:\n{matrix_A}\nMatrix U:\n{U}\nMatrix E:\n{E}\nMatrix V^T:\n{V_T}\n")

checked_A = np.dot(np.dot(U, E), V_T)
print(f"Checked matrix A:\n{checked_A}\n")

# task 2
import pandas as pd

# Зчитування CSV файлу
file_path = 'ratings.csv'
df = pd.read_csv(file_path)

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

ratings_matrix = ratings_matrix.dropna(thresh=200, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=100, axis=1)

ratings_matrix_filled = ratings_matrix.fillna(2.5)
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k=3)

users_to_visualize = 20
U_3d = U[:users_to_visualize, :3]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot each user as a point in 3D space
for i in range(users_to_visualize):
    ax.scatter(U_3d[i, 0], U_3d[i, 1], U_3d[i, 2])

ax.set_title('Users')
plt.show()

films_to_visualize = 20
R_subset = R[:films_to_visualize, :]

Vt_3d = R_subset.T

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot each user as a point in 3D space
for i in range(films_to_visualize):
    ax.scatter(Vt_3d[0, i], Vt_3d[1, i], Vt_3d[2, i])

ax.set_title('Films')
plt.show()
