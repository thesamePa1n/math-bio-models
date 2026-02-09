import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

L = 100.0
D = 32.0
gamma = 0.005545
alpha = 4.37e-4
beta = 4.37e-3
U_th = 70e-9
T = 10.0
h = 0.1
tau = 1

N = int(L / h) + 1
M = int(T / tau) + 1

x = np.linspace(0, L, N)
t = np.linspace(0, T, M)

centers = [20, 50, 80]

a = -2 * D * tau / h**2
c = a
b = 3 + 4 * D * tau / h**2 + 2 * gamma * tau

n = N - 2
ab = np.zeros((3, n))
ab[0, 1:] = a * np.ones(n - 1)  # Верхняя диагональ
ab[1, :] = b * np.ones(n)       # Главная диагональ
ab[2, :-1] = c * np.ones(n - 1) # Нижняя диагональ

U = np.zeros((M, N))
U[:, 0] = 0
U[:, -1] = 0

for j in range(M - 1):
    U_curr = U[j]
    F = np.zeros(N)
    for center in centers:
        idx = int(round(center / h))
        Uc = U_curr[idx]
        powered_U = Uc ** 2.5
        powered_Uth = U_th ** 2.5
        f = alpha + beta * powered_U / (powered_Uth + powered_U)
        F[idx] += f / h
    rhs = np.zeros(n)
    if j == 0:
        U_prev = np.zeros(N)
    else:
        U_prev = U[j - 1]
    for k in range(n):
        i = k + 1
        rhs[k] = 4 * U_curr[i] - U_prev[i] + 2 * tau * F[i]
    V = solve_banded((1, 1), ab, rhs)
    U[j + 1, 1:-1] = V

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, TT = np.meshgrid(x, t)
ax.plot_surface(X, TT, U, cmap='jet')
ax.set_xlabel('x, мкм')
ax.set_ylabel('t, час')
ax.set_zlabel('U, моль/л')
plt.show()