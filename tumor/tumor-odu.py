import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Функция системы ОДУ
def predator_prey(z, t, lambda1, beta1, c, lambda2, beta2):
    x, y = z
    if y < 0: y = 0  # Избежать отрицательных значений
    y23 = y**(2/3)
    dxdt = (-lambda1 + beta1 * y23 / (1 + x) * (1 - x / c)) * x
    dydt = lambda2 * y - beta2 * x * y23 / (1 + x)
    return [dxdt, dydt]

# Параметры из документа
lambda1 = 1.0
lambda2 = 1.0
beta1 = 1.0
c = 3.0
beta2_values = [3.0, 3.48, 5.0]  # Варьируем β2
z0 = [5, 0.5]  # x(0)=0.5, y(0)=1.0
t = np.linspace(0, 10, 101)  # Время от 0 до 10

# Построение графиков
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, beta2 in enumerate(beta2_values):
    sol = odeint(predator_prey, z0, t, args=(lambda1, beta1, c, lambda2, beta2))
    axs[i].plot(t, sol[:, 0], 'r', label='Лимфоциты (x)')
    axs[i].plot(t, sol[:, 1], 'b', label='Опухоль (y)')
    axs[i].set_title(f'β₂ = {beta2}')
    axs[i].set_xlabel('Время (t)')
    axs[i].set_ylabel('Плотность')
    axs[i].legend()
    axs[i].grid(True)
plt.tight_layout()
plt.show() 