from brian2 import *

start_scope()

# =========================
# Simulation parameters
# =========================
dt = 0.02*ms
defaultclock.dt = dt
duration = 200*ms

N = 3
g_gap = 0.05*msiemens/cm**2

# =========================
# Hodgkin–Huxley constants
# =========================
Cm  = 1*uF/cm**2
gNa = 120*msiemens/cm**2
gK  = 36*msiemens/cm**2
gL  = 0.3*msiemens/cm**2

ENa = 50*mV
EK  = -77*mV
EL  = -54.387*mV

# =========================
# HH neuron equations
# =========================
eqs = '''
dv/dt = (Iext - INa - IK - IL + Igap) / Cm : volt

INa = gNa*m**3*h*(v-ENa) : amp/meter**2
IK  = gK*n**4*(v-EK)     : amp/meter**2
IL  = gL*(v-EL)          : amp/meter**2

dm/dt = alpham*(1-m) - betam*m : 1
dh/dt = alphah*(1-h) - betah*h : 1
dn/dt = alphan*(1-n) - betan*n : 1

alpham = 0.1/mV*(v+40*mV)/(1-exp(-(v+40*mV)/(10*mV)))/ms : Hz
betam  = 4*exp(-(v+65*mV)/(18*mV))/ms : Hz

alphah = 0.07*exp(-(v+65*mV)/(20*mV))/ms : Hz
betah  = 1/(1+exp(-(v+35*mV)/(10*mV)))/ms : Hz

alphan = 0.01/mV*(v+55*mV)/(1-exp(-(v+55*mV)/(10*mV)))/ms : Hz
betan  = 0.125*exp(-(v+65*mV)/(80*mV)) /ms : Hz

Iext : amp/meter**2
Igap : amp/meter**2
'''

neurons = NeuronGroup(
    N,
    eqs,
    method='exponential_euler'
)

# =========================
# Initial conditions
# =========================
neurons.v = -65*mV + 0.5*mV*randn(N)

neurons.m = 'alpham/(alpham+betam)'
neurons.h = 'alphah/(alphah+betah)'
neurons.n = 'alphan/(alphan+betan)'

neurons.Iext = 0*amp/meter**2

# =========================
# Gap junction coupling
# Igap_i = g_gap * sum_j (Vj - Vi)
# =========================
syn = Synapses(
    neurons, neurons,
    model='Igap_post = g_gap*(v_pre - v_post) : amp/meter**2 (summed)'
)
syn.connect(condition='i != j')

# =========================
# External current stimulus
# =========================
@network_operation(dt=dt)
def stimulus():
    if defaultclock.t < 5*ms:
        neurons.Iext[0] = 10*uA/cm**2
    else:
        neurons.Iext[0] = 0*uA/cm**2

# =========================
# Monitors
# =========================
M = StateMonitor(neurons, 'v', record=True)

# =========================
# Run
# =========================
run(duration)

# =========================
# Plot
# =========================
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
for i in range(N):
    plt.plot(M.t/ms, M.v[i]/mV, label=f'neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('V (mV)')
plt.title('HH 3-neuron gap junction coupling (Brian2)')
plt.legend()
plt.show()
