import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

um, KL, Y, Ki = 3.36, 20.0, 0.5, 0.7
X0, L0 = 0.01, 1.5

def haldane(t, y):
    X, L = y
    mu = um * L / (KL + L + L**2/Ki) 
    return [mu * X, -mu * X / Y]

t_span = (0, 60)
t_eval = np.linspace(0, 60, 500)
sol = solve_ivp(haldane, t_span, [X0, L0], t_eval=t_eval, method='RK45')

plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], 'r-', linewidth=2, label='Микроорганизмы (X)')
plt.plot(sol.t, sol.y[1], 'b-', linewidth=2, label='Субстрат (L)')
plt.xlabel('Время, сут')
plt.ylabel('Концентрация, мг/л')
plt.title('Модель Халдейна: очистка сточных вод')
plt.legend()
plt.grid(alpha=0.3)
plt.show()