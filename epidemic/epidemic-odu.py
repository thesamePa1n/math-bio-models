import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def sis(y, t, beta, gamma, mu):
    s, i = y
    dsdt = -beta * s * i + mu * (1 - s) + gamma * i
    didt = beta * s * i - gamma * i - mu * i
    return [dsdt, didt]

beta = 0.5  # Коэффициент передачи инфекции
gamma = 0.1  # Коэффициент выздоровления
mu = 0.001  # Коэффициент рождаемости/смертности
y0 = [0.99, 0.01]  # Начальные условия: S(0)=0.99, I(0)=0.01
t = np.linspace(0, 100, 101) 

sol = odeint(sis, y0, t, args=(beta, gamma, mu))

plt.plot(t, sol[:, 0], 'b', label='Восприимчивые (S)')
plt.plot(t, sol[:, 1], 'r', label='Инфицированные (I)')
plt.legend()
plt.xlabel('Время (t)')
plt.ylabel('Доля популяции')
plt.title('Модель SIS с рождаемостью и смертностью')
plt.grid(True)
plt.show()  