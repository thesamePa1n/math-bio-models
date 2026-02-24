import numpy as np
import matplotlib.pyplot as plt

def f1(u, uu, v):
  return u * (2.4-u-6*uu-4*v)

def f2(u, uu, v):
  return uu * (4-uu-u-10*v)

def g1(u, uu, v):
  return -v * (1.0-0.25*u-4*uu+v)

N = 100
D1 = 0.0001
D2 = 0.0002
D3 = 0.0001
x = np.zeros(N+1)
alpha = np.zeros(N+1)
beta = np.zeros(N+1)
c1, c2, c3 = 0, 0, 0
u1 = np.zeros((N+1, N+1))
u2 = np.zeros((N+1, N+1))
v1 = np.zeros((N+1, N+1))
tau = 0.09
h = 1.0 / N
a1 = D1*tau/(h**2)
b1 = a1

for i in range(N+1):
  x[i] = i*h
  u1[i][0] = 5 * (x[i]-0.5)**2 + 0.2
  u2[i][0] = 3 * (x[i]-0.5)**2 + 0.05
  v1[i][0] = (x[i]-0.5)**2 + 0.3
  
for j in range(N):
  alpha[1] = 0
  beta[1] = 0
  u1[0][j+1] = 1.45
  u2[0][j+1] = 0.8
  v1[0][j+1] = 0.55
  
  for i in range(1, N):
    c1 = 2*a1 + 1
    alpha[i+1] = b1 / (c1 - a1*alpha[i])
    d = u1[i][j] + f1(u1[i][j], u2[i][j], v1[i][j]) * tau
    beta[i+1] = (a1*beta[i]+d) / (c1-a1*alpha[i])
    
  u1[N][j+1] = beta[N] / (1 - alpha[N])
  
  for i in range(N-1, 0, -1):
    u1[i][j+1] = u1[i+1][j+1]*alpha[i+1]+beta[i+1]
  
  a2 = D2*tau / (h**2)
  b2 = a2
  alpha[1] = 0
  beta[1] = 0
  
  for i in range(1, N):
    c2 = 2*a2 + 1
    alpha[i+1] = b2 / (c2 - a2*alpha[i])
    d = u2[i][j]+f2(u1[i][j], u2[i][j], v1[i][j])*tau
    beta[i+1] = (a2*beta[i]+d) / (c2-a2*alpha[i])
    
  u2[N][j+1] = beta[N] / (1 - alpha[N])
  
  for i in range(N-1, 0, -1):
    u2[i][j+1] = u2[i+1][j+1]*alpha[i+1] + beta[i+1]
    
  a3 = D3*tau / (h**2)
  b3 = a3
  alpha[1] = 0
  beta[1] = 0.5
  
  for i in range(1, N):
    c3 = 2*a3 + 1
    alpha[i+1] = b3 / (c3 - a3*alpha[i])
    d = v1[i][j] + g1(u1[i][j], u2[i][j], v1[i][j])*tau
    beta[i+1] = (a3*beta[i]+d) / (c3 - a3*alpha[i])
    
  v1[N][j+1] = beta[N] / (1 - alpha[N])
  
  for i in range(N-1, 0, -1):
    v1[i][j+1] = v1[i+1][j+1]*alpha[i+1]+beta[i+1]
  
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(range(N), u1[1:, 0], color="blue", label="u1(x, t=0)")
axes[0].plot(range(N), u2[1:, 0], color="green", label="u2(x, t=0)")
axes[0].plot(range(N), v1[1:, 0], color="red", label="v(x, t=0)")
axes[0].legend()

axes[1].plot(range(N), u1[1:, 10], color="blue", label="u1(x, t=10)")
axes[1].plot(range(N), u2[1:, 10], color="green", label="u2(x, t=10)")
axes[1].plot(range(N), v1[1:, 10], color="red", label="v(x, t=10)")
axes[1].legend()

axes[2].plot(range(N), u1[1:, 20], color="blue", label="u1(x, t=20)")
axes[2].plot(range(N), u2[1:, 20], color="green", label="u2(x, t=20)")
axes[2].plot(range(N), v1[1:, 20], color="red", label="v(x, t=20)")

plt.tight_layout()
plt.legend()
plt.show()