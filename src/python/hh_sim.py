import numpy as np
import math

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Hodgkin-Huxley parameters (standard)
C_m = 1.0  # uF/cm^2
g_Na = 120.0
g_K = 36.0
g_L = 0.3
E_Na = 50.0
E_K = -77.0
E_L = -54.387

def alpha_m(V):
    # α_m(V) = 0.1 * (V + 40) / (1 - exp(-(V + 40) / 10))
    # Rate constant for m-gate opening (Na+ activation)
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

def beta_m(V):
    # β_m(V) = 4 * exp(-(V + 65) / 18)
    # Rate constant for m-gate closing (Na+ activation)
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V):
    # α_h(V) = 0.07 * exp(-(V + 65) / 20)
    # Rate constant for h-gate opening (Na+ inactivation)
    return 0.07 * np.exp(-(V + 65.0) / 20.0)

def beta_h(V):
    # β_h(V) = 1 / (1 + exp(-(V + 35) / 10))
    # Rate constant for h-gate closing (Na+ inactivation)
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

def alpha_n(V):
    # α_n(V) = 0.01 * (V + 55) / (1 - exp(-(V + 55) / 10))
    # Rate constant for n-gate opening (K+ activation)
    return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

def beta_n(V):
    # β_n(V) = 0.125 * exp(-(V + 65) / 80)
    # Rate constant for n-gate closing (K+ activation)
    return 0.125 * np.exp(-(V + 65.0) / 80.0)

def I_ion(V, m, h, n):
    # Total ionic current: I_ion = I_Na + I_K + I_L
    #
    # I_Na = g_Na * m³ * h * (V - E_Na)   [Sodium current]
    # I_K  = g_K  * n⁴ * (V - E_K)        [Potassium current]
    # I_L  = g_L  * (V - E_L)             [Leak current]
    #
    # where m, h, n are gating variables (0 to 1)
    I_Na = g_Na * (m ** 3) * h * (V - E_Na)
    I_K = g_K * (n ** 4) * (V - E_K)
    I_L = g_L * (V - E_L)
    return I_Na + I_K + I_L

def derivatives(state, t, I_ext, g_gap, V_array):
    """
    Compute derivatives of all state variables for a single neuron.

    Hodgkin-Huxley ODEs:
      C_m * dV/dt = I_ext - I_ion + I_coupling
      dm/dt = α_m(V)*(1 - m) - β_m(V)*m
      dh/dt = α_h(V)*(1 - h) - β_h(V)*h
      dn/dt = α_n(V)*(1 - n) - β_n(V)*n

    Gap-junction coupling (diffusive, all-to-all):
      I_coupling = g_gap * Σ_j (V_j - V_i)
                 = g_gap * (Σ_j V_j  -  N * V_i)
    """
    V, m, h, n = state

    # I_coupling = g_gap * Σ_j (V_j - V_i) = g_gap * (sum(V_all) - N*V_i)
    coupling = g_gap * (np.sum(V_array) - V * V_array.size)

    # dV/dt = (I_ext - I_ion + I_coupling) / C_m
    dVdt = (I_ext - I_ion(V, m, h, n) + coupling) / C_m

    # dm/dt = α_m(V)*(1 - m) - β_m(V)*m
    dm = alpha_m(V) * (1.0 - m) - beta_m(V) * m

    # dh/dt = α_h(V)*(1 - h) - β_h(V)*h
    dh = alpha_h(V) * (1.0 - h) - beta_h(V) * h

    # dn/dt = α_n(V)*(1 - n) - β_n(V)*n
    dn = alpha_n(V) * (1.0 - n) - beta_n(V) * n

    return np.array([dVdt, dm, dh, dn])

def rk4_step(states, dt, I_exts, g_gap):
    """
    Fourth-order Runge-Kutta (RK4) integrator for one time step.

    Given dy/dt = f(t, y), RK4 advances the solution as:
        k1 = f(t,       y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt,   y + dt   * k3)

        y(t + dt) = y(t) + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    Here y = [V, m, h, n] for each neuron.
    Coupling uses the *current* voltage snapshot (explicit method).
    """
    N = states.shape[0]
    V_array = states[:,0]  # snapshot of all membrane potentials

    k1 = np.zeros_like(states)
    k2 = np.zeros_like(states)
    k3 = np.zeros_like(states)
    k4 = np.zeros_like(states)

    # k1 = f(t, y)
    for i in range(N):
        k1[i] = derivatives(states[i], 0, I_exts[i], g_gap, V_array)

    # k2 = f(t + dt/2, y + dt/2 * k1)
    mid1 = states + 0.5 * dt * k1
    for i in range(N):
        k2[i] = derivatives(mid1[i], 0, I_exts[i], g_gap, V_array)

    # k3 = f(t + dt/2, y + dt/2 * k2)
    mid2 = states + 0.5 * dt * k2
    for i in range(N):
        k3[i] = derivatives(mid2[i], 0, I_exts[i], g_gap, V_array)

    # k4 = f(t + dt, y + dt * k3)
    end = states + dt * k3
    for i in range(N):
        k4[i] = derivatives(end[i], 0, I_exts[i], g_gap, V_array)

    # y(t + dt) = y(t) + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    return states + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate(N=3, tmax=100.0, dt=0.01, g_gap=0.1):
    """
    Run a coupled Hodgkin-Huxley network simulation.

    Parameters:
        N     : number of neurons
        tmax  : total simulation time (ms)
        dt    : integration time step (ms)
        g_gap : gap-junction conductance (mS/cm²)

    Initial conditions:
        V(0)  ≈ -65 mV  (resting potential with small noise)
        m(0)  = m_∞(V)  = α_m / (α_m + β_m)   [steady-state]
        h(0)  = h_∞(V)  = α_h / (α_h + β_h)
        n(0)  = n_∞(V)  = α_n / (α_n + β_n)
    """
    steps = int(tmax / dt)

    # State matrix: each row is [V, m, h, n] for one neuron
    states = np.zeros((N, 4))

    # V(0) ≈ -65 mV + small noise
    states[:, 0] = -65.0 + 0.5 * np.random.randn(N)

    # Gating variables initialized to steady-state values:
    #   x_∞(V) = α_x(V) / (α_x(V) + β_x(V))   for x ∈ {m, h, n}
    states[:, 1] = alpha_m(states[:, 0]) / (alpha_m(states[:, 0]) + beta_m(states[:, 0]))
    states[:, 2] = alpha_h(states[:, 0]) / (alpha_h(states[:, 0]) + beta_h(states[:, 0]))
    states[:, 3] = alpha_n(states[:, 0]) / (alpha_n(states[:, 0]) + beta_n(states[:, 0]))

    I_exts = np.zeros(N)
    # apply a brief current to neuron 0
    I_exts[0] = 10.0

    traj = np.zeros((steps, N))
    for s in range(steps):
        t = s * dt
        # turn off current after 5 ms
        if t > 5.0:
            I_exts[0] = 0.0
        states = rk4_step(states, dt, I_exts, g_gap)
        traj[s,:] = states[:,0]
    return traj, dt

if __name__ == '__main__':
    traj, dt = simulate(N=3, tmax=200.0, dt=0.02, g_gap=0.05)
    if plt is not None:
        t = np.arange(traj.shape[0]) * dt
        for i in range(traj.shape[1]):
            plt.plot(t, traj[:,i], label=f'neuron {i}')
        plt.xlabel('time (ms)')
        plt.ylabel('V (mV)')
        plt.legend()
        plt.title('HH 3-neuron coupling (Python)')
        plt.show()
    else:
        print('Simulation complete. Install matplotlib to see plots.')
