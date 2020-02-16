import matplotlib.pyplot as plt
from Spike.config import *
from Spike.RLS import FORCESPIKE

np.random.seed(seed)


class Population(object):
    def __init__(self, N, sp):
        self.N = N
        self.target = target
        self.sparsity = sp
        self.nouputs = self.target.shape[0]
        self.rls = FORCESPIKE()
        self.init_network()
        self.inputscurr = input

    def init_network(self):
        self.alpha = dt * 0.1
        self.Pinv = np.identity(self.N) * self.alpha
        self.IPSC = np.zeros((self.N, 1))  # post-synaptic current
        self.h = np.zeros((self.N, 1))  # filtered firing rate
        self.r = np.zeros((self.N, 1))  # second storage for filtered firing rate
        self.hr = np.zeros((self.N, 1))  # third storage
        self.JD = 0 * self.IPSC  # storage for each spike time
        self.tspike = np.zeros((4 * nt, 2))  # storage for spike times
        self.ns = 0  # no. of spike
        self.z = np.zeros((self.nouputs, 1))  # initial output
        self.v = vreset + np.random.rand(self.N, 1) * (30 - vreset)  # voltage initial.
        self.v_ = self.v  # voltage at previous timestep
        self.RECB = np.zeros((nt, 10))  # weight storage (subset)
        self.OMEGA = G * np.multiply(np.random.randn(self.N, self.N), np.random.rand(self.N, self.N)) / (
                np.sqrt(N) * self.sparsity)  # initial weight matrix with fixed random weights
        self.BPhi = np.zeros((self.N, self.nouputs))  # initial weight matrix learned with FORCE

        # set row average weight to zero explicitly
        for i in range(self.N):
            QS = np.where(np.abs(self.OMEGA[i, :]) > 0)[0]
            self.OMEGA[i, QS] = self.OMEGA[i, QS] - np.sum(self.OMEGA[i, QS]) / len(QS)

        self.E = (2 * np.random.rand(self.N, self.nouputs) - 1) * Q
        self.REC2 = np.zeros((nt, 20))
        self.REC = np.zeros((nt, 10))
        self.current = np.zeros((self.nouputs, nt))  # output current
        self.tlast = np.zeros((self.N, 1))  # to set for refractory times
        self.BIAS = vpeak  # BIAS current, can help decrease/increase firing, 0 is fine

    def update_v_get_spikes(self, i):
        I = self.IPSC + self.E * self.z + self.BIAS + np.repeat(self.inputscurr[:,i].reshape(-1,1),self.N,0)
        dv = (dt * (i + 1) > (self.tlast + tref)) * (-self.v + I) / tm  # voltage equation
        self.v = self.v + dt * dv
        index = np.where(self.v >= vpeak)[0].reshape(-1, 1)  # neurons that reached threshold
        return index

    def calculate_JD(self, index, i):
        # store spike times and get the weight matrix column sum of spikes
        if len(index) > 0:
            self.JD = np.sum(self.OMEGA[:, index], 1).reshape(-1, 1)  # compute increase in current due to spiking
            self.tspike[self.ns: self.ns + len(index), :] = np.hstack((index, 0 * index + dt * (i + 1)))
            self.ns += len(index)  # total number of spikes so far

    def update_firing_rate(self, index):
        # Code if the rise time is 0, and if the rise time is positive
        if tr == 0:  # exponential firing rate
            self.IPSC = self.IPSC * np.exp(-dt / td) + self.JD * (len(index) > 0) / td
            self.r = self.r * np.exp(-dt / td) + (self.v >= vpeak) / td
        else:  # double exponential firing rate
            self.IPSC = self.IPSC * np.exp(-dt / tr) + self.h * dt
            self.h = self.h * np.exp(-dt / td) + self.JD * (len(index) > 0) / (tr * td)

            self.r = self.r * np.exp(-dt / tr) + self.hr * dt
            self.hr = self.hr * np.exp(-dt / td) + 1 * (self.v >= vpeak) / (tr * td)

    def record_varialbe(self, i):
        # this is options
        self.current[:, i] = self.z[:, 0]
        self.RECB[i, :] = self.BPhi[:10, :].transpose()
        self.REC2[i, :] = self.r[:20, :].transpose()
        self.REC[i, :] = self.v[:10, :].transpose()  # record voltage
        self.v = self.v + (vreset - self.v) * (self.v >= vpeak)

    def simulate_network(self):
        for i in range(nt):
            index = self.update_v_get_spikes(i)

            # update spikecpount and estimate weighted input firing rate
            self.calculate_JD(index, i)

            # update latest spiketimes
            self.tlast += (dt * (i + 1) - self.tlast) * (self.v >= vpeak)  # used to set refractory period of last spike

            # update firing rate: self.r
            self.update_firing_rate(index)

            # implement RLMS with FORCE
            self.z = np.matmul(self.rls.BPhi.transpose(), self.r)  # approximate outpu
            err = self.z - self.target[:, i]  # error
            ##RLMS
            if np.mod(i, step) == 0:
                if i > imin:
                    if i < icrit:
                        self.rls.update(self.r, err)

            self.v = self.v + (30 - self.v) * (self.v >= vpeak)

            self.record_varialbe(i)

            # reset
            self.v = self.v + (vreset - self.v) * (self.v >= vpeak)

            if np.mod(i, round(0.05 / dt)) == 1:
                plt.plot(self.target[0, :])
                plt.plot(self.current[0, :])
                # plt.show()
                plt.savefig("images/output_{}.png".format(i))
                # time.sleep(0.1)
                plt.close()


if __name__ == '__main__':
    pop = Population(N, sp)
    pop.simulate_network()
