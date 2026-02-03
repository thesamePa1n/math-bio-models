import numpy as np
import matplotlib.pyplot as plt


def progonka(a, b, c, d):
    n = len(d)
    cp = np.zeros(n-1)
    dp = np.zeros(n)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n-1):
        denom = b[i] - a[i-1] * cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i-1] * dp[i-1]) / denom

    dp[-1] = (d[-1] - a[-2] * dp[-2]) / (b[-1] - a[-2] * cp[-2])

    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in reversed(range(n-1)):
        x[i] = dp[i] - cp[i] * x[i+1]

    return x

def implicit_diffusion(u, d, dt, dx):
    if d == 0.0:
        return u.copy()

    r = d * dt / dx**2
    n = len(u)

    a = -r * np.ones(n-1)
    b = (1 + 2*r) * np.ones(n)
    c = -r * np.ones(n-1)

    c[0]  = -2*r
    a[-1] = -2*r

    return progonka(a, b, c, u)

Pi   = 3.3e-5 # интенсивность притока восприимчивых
beta = 0.75 # коэффициент передачи инфекции
mu   = 3.4e-5 # естественная смертность
l_rrisk  = 0.38 # пониженный риск инфицирования у диагностированных
kappa= 0.33 # переход из E в I
q    = 0.1 # заразность экспонированных
alpha= 0.33 # переход из I в J
gamma1   = 0.125 # скорость выздоровления инфицированных
gamma2   = 0.2 # скорость выздоровления диагностированных
delta= 0.006 # смертность, вызванная заболеванием

d1, d2, d3, d4, d5 = 0.025, 0.01, 0.001, 0.0, 0.0 #коэффициенты диффузии
dx = 0.1
dt = 0.03

for k, dk in enumerate([d1, d2, d3, d4, d5], start=1):
    r = dk * dt / dx**2
    if r > 0.5 + 1e-12:
        print("Условие устойчивости нарушено")

x = np.arange(-2.0, 2.0 + dx/2, dx)
nx = x.size

S = 0.98 * (np.exp(-5*x**2)) ** 3
E = np.zeros_like(x)
I = np.zeros_like(x)
mask = (-0.4 <= x) & (x <= 0.4)
I[mask] = 0.02
J = np.zeros_like(x)
R = np.zeros_like(x)

t_end = 20.0
nt = int(round(t_end / dt))

snap_times = [0, 5, 10, 15, 20]
snap_steps = {int(round(T/dt)): T for T in snap_times}
snaps = {name: [] for name in ["t", "S", "E", "I", "J", "R"]}

for n in range(nt + 1):
    if n in snap_steps:
        snaps["t"].append(snap_steps[n])
        snaps["S"].append(S.copy())
        snaps["E"].append(E.copy())
        snaps["I"].append(I.copy())
        snaps["J"].append(J.copy())
        snaps["R"].append(R.copy())

    if n == nt:
        break
    N = S + E + I + J + R
    N = np.maximum(N, 1e-12)

    infection = beta * (I + q*E + l_rrisk*J) / N

    S_half = S + dt * (-infection*S - mu*S + Pi)
    E_half = E + dt * ( infection*S - (mu + kappa)*E)
    I_half = I + dt * ( kappa*E - (mu + alpha + gamma1 + delta)*I)
    J_half = J + dt * ( alpha*I - (mu + gamma2 + delta)*J)
    R_half = R + dt * ( gamma1*I + gamma2*J - mu*R)

    S_half = np.maximum(S_half, 0.0)
    E_half = np.maximum(E_half, 0.0)
    I_half = np.maximum(I_half, 0.0)
    J_half = np.maximum(J_half, 0.0)
    R_half = np.maximum(R_half, 0.0)

    S = implicit_diffusion(S_half, d1, dt, dx)
    E = implicit_diffusion(E_half, d2, dt, dx)
    I = implicit_diffusion(I_half, d3, dt, dx)
    J = implicit_diffusion(J_half, d4, dt, dx)
    R = implicit_diffusion(R_half, d5, dt, dx)

fig, axs = plt.subplots(3, 2, figsize=(10, 6))
axs = axs.flatten()

variables = ["S", "E", "I", "J", "R"]
titles = [
    "Восприимчивые S(x,t)",
    "Экспонированные E(x,t)",
    "Инфицированные I(x,t)",
    "Диагностированные J(x,t)",
    "Выздоровевшие R(x,t)"
]

for i, var in enumerate(variables):
    for k, T in enumerate(snaps["t"]):
        axs[i].plot(x, snaps[var][k], label=f"t = {T}")
    axs[i].set_title(titles[i])
    axs[i].set_xlabel("x")
    axs[i].set_ylabel(var)
    axs[i].grid(True)
    axs[i].legend(fontsize=8)

axs[-1].axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()