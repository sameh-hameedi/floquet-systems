"""
This code solves a two-level Dirac PDE system with using a
Crank-Nicolson method.

The code also includes an optional 'edge' – a smooth function
varying between [-1,1] to simulate a defect around the origin
of the spatial domain.

The paramerers for governing the domain, time-stepping,
initial conditions and driving can be edited in the __init__ funtion.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import spsolve

class DiracSystem:

    def __init__(self):
        ### Domain parameters ###
        self.N = 200 # Number of nodes
        self.xmin = -10
        self.xmax = 10

        ### Time parameters ###
        self.tmax = 30
        self.dt = 0.09 # Timestep

        ### Wavepacket parameters ###
        self.x0 = 0 # Initial wavepacket centre
        self.k0 = 0 # Initial wavepacket momentum

        ### Forcing parameters ###
        self.A = 0 # Forcing amplitude
        self.w = 2*np.pi # Forcing angular frequency

        ### Edge parameters ###
        self.theta = 10 # Edge coefficent

        self.init_domain()
        self.init_vectors()
        self.init_params()
        self.init_gradient_matrices()
        self.init_edge_matrix()


    def gaussian(self):
        return np.exp(-1*(self.x - self.x0)**2 - 1j*self.k0*self.x)


    def edge(self):
        return self.theta * np.tanh(self.x)


    def init_domain(self):
        self.dx = (self.xmax - self.xmin) / self.N
        self.x = np.arange(self.xmin - self.dx, self.xmax + 2*self.dx, self.dx)
        self.lenx = len(self.x)


    def init_vectors(self):
        u10 = self.gaussian()
        u20 = np.zeros(self.lenx, dtype=complex)
        U0 = np.append(u10, u20)
        self.U = U0.copy()
        self.Utp1 = U0.copy()


    def init_params(self):
        self.nsteps = round(self.tmax/self.dt)
        self.alpha = 1/(2*self.dx)
        self.r = self.dt*self.alpha/2


    def init_gradient_matrices(self):
        nums = self.r * np.ones(self.lenx - 1)
        Dplus_data = np.concatenate([nums, -1*nums, np.ones(self.lenx)])
        Dminus_data = np.concatenate([-1*nums, nums, np.ones(self.lenx)])
        row = np.concatenate([np.arange(0, self.lenx-1, 1), np.arange(1, self.lenx, 1), np.arange(0, self.lenx, 1)])
        col = np.concatenate([np.arange(1, self.lenx, 1), np.arange(0, self.lenx-1, 1), np.arange(0, self.lenx, 1)])
        self.Dplus = coo_matrix((Dplus_data, (row, col)), shape=(self.lenx, self.lenx))
        self.Dminus = coo_matrix((Dminus_data, (row, col)), shape=(self.lenx, self.lenx))


    def init_edge_matrix(self):
        K_data = (-1j/2) * self.dt * self.edge()
        self.K = coo_matrix((K_data, (np.arange(0,self.lenx,1), np.arange(0,self.lenx,1))), shape=(self.lenx, self.lenx))


    def crank_nicolson_matrices(self):
        A = vstack([hstack([self.Dminus, self.K]), hstack([self.K, self.Dplus])])
        B = vstack([hstack([self.Dplus, self.K]), hstack([ self.K, self.Dminus])])
        return A.tocsr(), B.tocsr()


    def solve_and_plot(self):
        t = 0

        for i in range(self.nsteps):
            plt.clf()

            ### Crank-Nicolson step ###
            A, B = self.crank_nicolson_matrices()
            self.Utp1 = spsolve(A, B * self.U)
            self.U = self.Utp1.copy()

            ### Plotting ###
            plt.plot(self.x, np.abs(self.U[:self.lenx]).real, 'bo-', lw=1, markersize=0, label=r"$|\alpha_1(x)|$")
            plt.plot(self.x, np.abs(self.U[self.lenx:]).real, 'go-', lw=1, markersize=0, label=r"$|\alpha_2(x)|$")
            plt.axis((self.xmin, self.xmax, 0, 1.1))
            plt.grid(True)
            plt.xlabel(r"$x$")
            plt.ylabel(r"$|\alpha(x)|$")
            plt.legend(loc=1, fontsize=12)
            plt.title("Time = %1.3f" % (t + self.dt))
            plt.pause(0.01)

            t += self.dt


def main():
    sim = DiracSystem()
    sim.solve_and_plot()
    plt.show()


if __name__ == "__main__":
    main()
