import numpy as np

# neurons
dt = 5e-5
tref = 2e-3
tm = 1e-2
vreset = -65
vpeak = -40

seed = 1
td = 2e-2
tr = 2e-2

# network
N = 2000
sp = 0.1
noutputs = 1

# target dynamics - sine wave
T = 15
imin = round(5 / dt)
icrit = round(10 / dt)
step = 50
nt = round(T / dt)
freq1 = 1
freq2 = 4
Q = 10
G = 0.04
target1 = np.sin(2 * np.pi * np.linspace(1, nt, nt) * dt * freq1).reshape(1, -1)
target2 = np.sin(2 * np.pi * np.linspace(1, nt, nt) * dt * freq1).reshape(1, -1)
target = np.vstack((target1, target2))
# input
input1 = 0.1 * np.ones_like(target1)
input2 = 0.4 * np.ones_like(target2)
inputs = np.vstack((input1, input2))

# print(b.shape)
