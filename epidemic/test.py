import numpy as np
import matplotlib.pyplot as plt

def sis_rhs(y, t, beta, gamma, mu):
    s, i = y
    dsdt = -beta * s * i + mu * (1 - s) + gamma * i
    didt = beta * s * i - gamma * i - mu * i
    return np.array([dsdt, didt])

def rk4_step(y, t, h, beta, gamma, mu):
    k1 = sis_rhs(y, t, beta, gamma, mu)
    k2 = sis_rhs(y + 0.5 * h * k1, t + 0.5 * h, beta, gamma, mu)
    k3 = sis_rhs(y + 0.5 * h * k2, t + 0.5 * h, beta, gamma, mu)
    k4 = sis_rhs(y + h * k3, t + h, beta, gamma, mu)
    
    return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

beta = 0.5      # Коэффициент передачи инфекции
gamma = 0.1     # Коэффициент выздоровления
mu = 0.001      # Коэффициент рождаемости/смертности
y0 = np.array([0.99, 0.01])   # Начальные условия: S(0)=0.99, I(0)=0.01

t_start = 0
t_end = 100
n_steps = 2000          
h = (t_end - t_start) / n_steps   

t_values = np.linspace(t_start, t_end, n_steps + 1)
sol = np.zeros((n_steps + 1, 2))
sol[0] = y0

for i in range(n_steps):
    sol[i+1] = rk4_step(sol[i], t_values[i], h, beta, gamma, mu)

plt.plot(t_values, sol[:, 0], 'b-', label='Восприимчивые (S)')
plt.plot(t_values, sol[:, 1], 'r-', label='Инфицированные (I)')
plt.xlabel('Время (t)')
plt.ylabel('Доля популяции')
plt.title('Модель SIS с рождаемостью и смертностью')
plt.legend()
plt.grid(True)
plt.show()
