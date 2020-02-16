from Spike.config import *


class FORCESPIKE(object):
    def __init__(self):
        self.alpha = dt * 0.1
        self.Pinv = np.identity(N) * self.alpha
        self.BPhi = np.zeros((N, noutputs))  # initial weight matrix learned with FORCE

    def update(self, r, err):
        cd = np.matmul(self.Pinv, r)
        self.BPhi = self.BPhi - np.matmul(cd, err.transpose())
        self.Pinv = self.Pinv - np.matmul(cd, cd.transpose()) / (1 + np.matmul(r.transpose(), cd))

    def get_Phi(self):
        return self.BPhi
