"""
This code solves a Schrodinger equation using a Crank-Nicolson time-stepping
scheme.

The paramerers for governing the domain, time-stepping,
initial conditions and driving can be edited in the __init__ funtion.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, hstack, vstack, identity
from scipy.sparse.linalg import spsolve
#from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Helvetica']})
#rc('text', usetex=True)

class schro_FDM:

    def __init__(self):
        ### Domain parameters ###
        self.Nodes = 10000 # Number of nodes
        self.xmin = -65
        self.xmax = 65

        ### Time parameters ###
        self.tmax = 5
        self.dt = 0.01 # Timestep

        ### Forcing parameters ###
        self.A = 0 # Forcing amplitude
        self.w = 0 # Forcing angular frequency

        ### Delta ###
        self.delta = 0.1

        self.init_domain()
        self.init_vectors()
        self.init_params()
        self.init_gradient_matrices()


    ### Problem-Specific Functions ###
    def domain_wall(self):
        return np.tanh(self.delta*self.x)

    def edge_state(self):
        return 1 / np.cosh(self.x)

    def forcing(self, t):
        return self.A * np.cos(self.w*t)

    def V(self):
        return np.cos(4*np.pi*self.x)

    def W(self):
        return np.cos(2*np.pi*self.x)


    ### Main Setup ###
    def init_domain(self):
        self.dx = (self.xmax - self.xmin) / self.Nodes
        self.x = np.arange(self.xmin - self.dx, self.xmax + 2*self.dx, self.dx)
        self.lenx = len(self.x)


    def init_vectors(self):
        Psi0 = self.edge_state()
        self.Psi = Psi0.copy()


    def init_params(self):
        self.nsteps = round(self.tmax/self.dt)
        self.id = identity(self.lenx, format="csr")


    def init_gradient_matrices(self):
        # Periodic BCs
        nums_ii = self.V() + self.delta*self.domain_wall()*self.W() + 2/(self.dx**2)*np.ones(self.lenx)
        nums_ij = (-1/(self.dx**2)) * np.ones(self.lenx-1)
        data = np.concatenate((nums_ii, nums_ij, nums_ij, [-1/(self.dx**2), -1/(self.dx**2)]))
        row = np.concatenate((np.arange(0,self.lenx,1), np.arange(0,self.lenx-1,1), np.arange(1,self.lenx,1), [0,self.lenx-1]))
        col = np.concatenate((np.arange(0,self.lenx,1), np.arange(1,self.lenx,1), np.arange(0,self.lenx-1,1), [self.lenx-1,0]))
        self.D = coo_matrix((data, (row, col)), shape=(self.lenx,self.lenx)).tocsr()


    def forcing_matrix(self, t):
        alpha = 1j * self.delta * self.forcing(t) / self.dx
        nums = alpha * np.ones(self.lenx-1)
        data = np.concatenate((nums, -1*nums, [-1*alpha, alpha]))
        row = np.concatenate((np.arange(0,self.lenx-1,1), np.arange(1,self.lenx,1), [0, self.lenx-1]))
        col = np.concatenate((np.arange(1,self.lenx,1), np.arange(0,self.lenx-1,1), [self.lenx-1, 0]))
        return coo_matrix((data, (row, col)), shape=(self.lenx,self.lenx)).tocsr()


    def solve_and_plot(self):
        t = 0
        time = np.zeros(self.nsteps)
        norm = np.zeros(self.nsteps)
        Psi = np.zeros((self.lenx,self.nsteps))

        for i in range(self.nsteps):
            plt.clf()

            ### Crank-Nicolson step ###
            F = self.forcing_matrix(t)
            mat = (1j * self.dt / 2) * (self.D + F)
            self.Psi = spsolve(self.id + mat, (self.id - mat) * self.Psi)
            Psi[:,i] = np.abs(self.Psi)

            ### Norm ###
            n = np.trapz(np.square(np.abs(self.Psi)), dx=self.dx)
            norm[i] = np.sqrt(n)
            print(n/norm[0])

            ### Time ###
            time[i] = t

            ### Plot solution ###
            plt.plot(self.x, np.abs(self.Psi), "bo-", lw=0.8, markersize=0, label=r"$|\Psi|$")
            plt.plot(self.x, self.domain_wall(), 'r--', lw=0.8)
            plt.axis((self.xmin, self.xmax, -1.5, 1.5))
            plt.grid(True)
            plt.xlabel(r"$x$", fontsize=15)
            plt.ylabel(r"$|\Psi(x)|$", fontsize=15)
            plt.legend(loc=1, fontsize=15)
            plt.title("Time = %1.3f" % (t + self.dt))
            plt.pause(0.01)

            t += self.dt

        return time, self.x, Psi


def main():
    #from mpl_toolkits import mplot3d
    #ax = plt.axes(projection ='3d')
    sim = schro_FDM()
    t, x, Psi = sim.solve_and_plot()
    #T, X = np.meshgrid(t, x)
    #ax.plot_surface(X, T, Psi)
    plt.show()


if __name__ == "__main__":
    main()
