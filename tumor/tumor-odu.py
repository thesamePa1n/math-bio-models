import numpy as np
import matplotlib.pyplot as plt

def f(x, y, l1, l2, b1, b2, c):
    y_safe = max(y, 0.0)

    dxdt = x * (-l1 + b1 * (y_safe**(2/3)) * ((1 - x/c) / (1 + x)))
    dydt = l2 * y - (b2 * x * (y_safe**(2/3))) / (1 + x)
    
    return dxdt, dydt

def rk4_step(x, y, h, params):
    k1, l1 = f(x, y, *params)
    k2, l2 = f(x + h/2*k1, y + h/2*l1, *params)
    k3, l3 = f(x + h/2*k2, y + h/2*l2, *params)
    k4, l4 = f(x + h*k3, y + h*l3, *params)
    
    x_next = x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    y_next = y + h/6 * (l1 + 2*l2 + 2*l3 + l4)
    return x_next, y_next

l1, l2, b1, c = 1.0, 1.0, 1.0, 3.0
beta2_list = [3.0, 3.48, 5.0] 
x0, y0 = 2.0, 5.0 
t_end, h = 2.5, 0.001 

fig, axes = plt.subplots(3, 1, figsize=(10, 6))

for i, b2 in enumerate(beta2_list):
    n_steps = int(t_end / h) + 1
    t = np.linspace(0, t_end, n_steps)
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    x[0], y[0] = x0, y0
    
    for n in range(n_steps - 1):
        x[n+1], y[n+1] = rk4_step(x[n], y[n], h, (l1, l2, b1, b2, c))
    
    axes[i].plot(t, x, color='red', label='Лимфоциты x(t)', linewidth=1.5)
    axes[i].plot(t, y, color='orange', label='Опухоль y(t)', linewidth=1.5)
    
    axes[i].set_title(f'Динамика при beta2 = {b2}')
    axes[i].set_xlabel('Время t')
    axes[i].set_ylabel('Концентрация')
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()
    axes[i].set_ylim(0, max(max(x), max(y)) * 1.1)

plt.tight_layout()
plt.show()
