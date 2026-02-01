from brian2 import *

start_scope()

# =========================
# Simulation parameters
# =========================
dt = 0.01*ms
defaultclock.dt = dt
duration = 50*ms

# =========================
# HH constants
# =========================
Cm  = 1*uF/cm**2
gNa = 120*msiemens/cm**2
gK  = 36*msiemens/cm**2
gL  = 0.3*msiemens/cm**2

ENa = 50*mV
EK  = -77*mV
EL  = -54.387*mV

# =========================
# HH equations
# =========================
eqs = '''
dv/dt = (Iext - INa - IK - IL) / Cm : volt

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
betan  = 0.125*exp(-(v+65*mV)/(80*mV))/ms : Hz

Iext : amp/meter**2
'''

# =========================
# Neuron
# =========================
neuron = NeuronGroup(1, eqs, method='exponential_euler')

# Initial conditions
neuron.v = -65*mV
neuron.m = 'alpham/(alpham+betam)'
neuron.h = 'alphah/(alphah+betah)'
neuron.n = 'alphan/(alphan+betan)'

# Inject constant current
neuron.Iext = 10*uA/cm**2

# =========================
# Monitor
# =========================
M = StateMonitor(neuron, 'v', record=True)

# =========================
# Run
# =========================
run(duration)

# =========================
# Plot
# =========================
import matplotlib.pyplot as plt

plt.plot(M.t/ms, M.v[0]/mV)
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Single Hodgkin–Huxley neuron (Brian2)")
plt.show()
