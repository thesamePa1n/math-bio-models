import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

a = 3
b = 1
c = 3
d = 1
k_x = 1
k_y = 1

def predator_prey(t, vars):
    x, y = vars
    dxdt = a*x - b*x*y - k_x*x
    dydt = (c*x*y)/(1 + d*x) - k_y*y
    return [dxdt, dydt]

x0 = 1.499
y0 = 1.923

t_span = (0, 50)
t_eval = np.linspace(0, 50, 2000)

sol = solve_ivp(predator_prey, t_span, [x0, y0], t_eval=t_eval)

plt.figure(figsize=(10,5))
plt.plot(sol.t, sol.y[0], label='x(t) — жертвы')
plt.plot(sol.t, sol.y[1], label='y(t) — хищники')
plt.xlabel('t')
plt.ylabel('Популяции')
plt.legend()
plt.grid(True)
plt.show()