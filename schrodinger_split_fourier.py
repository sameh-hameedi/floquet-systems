import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, hstack, vstack
from scipy.sparse.linalg import spsolve, expm
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift


class Schrodinger:

    def __init__(self):
        ### Domain parameters ###
        self.N = 2**16 # Number of nodes / 2
        self.dx = 1e-3

        ### Time parameters ###
        self.tmax = 5
        self.dt = 0.01

        ### Forcing parameters ###
        self.A = 0 # Forcing amplitude
        self.w = 0 # Forcing angular frequency

        ### PDE parameters ###
        self.delta = 0.1 # Coefficent

        self.init_domain()
        self.init_momenta()
        self.init_vectors()
        self.spatial_operator()


    ### Helper Functions ###
    def norm(self, v):
        return np.sqrt(abs(np.vdot(v,v)))

    def gaussian(self, width):
        return np.exp(-self.x**2 / 2*width) / (np.sqrt(2*np.pi)*width)


    ### Problem-Specific Functions ###
    def edge_state(self):
        return 1 / np.cosh(self.x)

    def domain_wall(self):
        return np.tanh(self.delta*self.x)

    def forcing(self, t):
        return self.A * np.cos(self.w*t)

    def V(self):
        return np.cos(4*np.pi*self.x)

    def W(self):
        return np.cos(2*np.pi*self.x)


    ### Main Setup ###
    def init_domain(self):
        self.xmax = self.N * self.dx
        self.x = np.arange(-1*self.xmax, self.xmax, self.dx)
        self.lenx = len(self.x)


    def init_momenta(self):
        self.kmax = 2 * np.pi / (self.N * self.dx)
        self.k = self.kmax * np.arange(-1*self.N/2, self.N/2, 1/2)


    def init_vectors(self):
        Psi0 = self.edge_state() 
        #Psi0 = Psi0 / self.norm(Psi0)
        self.Psi = Psi0.copy()


    def spatial_operator(self):
        H_r = self.V() + self.delta * self.domain_wall() * self.W()
        self.opr_r = np.exp(-1j*H_r*self.dt/2)


    def opr_k(self, t):
        H_k = np.square(self.k) - 2 * self.delta * self.forcing(t) * self.k
        return np.exp(-1j*H_k*self.dt)


    def solve_and_plot(self):
        t = 0
        nsteps = round(self.tmax/self.dt)

        norm_ = np.zeros(nsteps)

        for i in range(nsteps):
            plt.clf()

            ### Half-step in real space ###
            #self.Psi = np.multiply(self.Psi, self.opr_r)

            ### Step in momentum space ###
            #self.Psi = np.multiply(fft(self.Psi), self.opr_k(t))

            ### Half-step in real space ###
            #self.Psi = np.multiply(ifft(self.Psi), self.opr_r)

            ### Half-step in real space ###
            self.Psi = np.multiply(self.Psi, self.opr_r)

            ### Step in momentum space ###
            self.Psi = np.multiply(fftshift(fft(self.Psi)/self.N), self.opr_k(t))

            ### Half-step in real space ###
            self.Psi = np.multiply(ifft(ifftshift(self.Psi))*self.N, self.opr_r)

            ### Norm ###
            norm = self.norm(self.Psi)
            norm_[i] = norm
            print(norm)

            ### Plot solution ###
            plt.plot(self.x, np.abs(self.Psi), "bo-", lw=0.8, markersize=0, label=r"$|\Psi(x)|$")
            plt.plot(self.x, self.domain_wall(), 'r--', lw=0.8)
            plt.axis((-1*self.xmax, self.xmax, -1.5, 1.5))
            plt.grid(True)
            plt.xlabel(r"$x$", fontsize=15)
            plt.ylabel(r"$|\Psi(x)|$", fontsize=15)
            plt.legend(loc=1, fontsize=15)
            plt.title("Time = %1.3f" % (t + self.dt))
            plt.pause(0.01)

            t += self.dt

def main():
    sim = Schrodinger()
    sim.solve_and_plot()
    plt.show()


if __name__ == "__main__":
    main()
