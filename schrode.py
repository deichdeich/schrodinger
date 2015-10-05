import numpy as np
from matplotlib import pyplot as pl
from matplotlib import animation
from scipy.fftpack import fft,ifft


class dinger(object):
    def __init__(self, x, init_psi, v_arr,
                 k_min = None, hbar=1, m=1, t_init=0.0):
        """
        x : position array
        init_psi : initial wave function position, array
        v_arr : array, defines V(x)
        k_min : minimum value of momentum
        """
        self.x, init_psi, self.v_arr = map(np.asarray, (x, init_psi, v_arr))
        N = self.x.size
        assert self.x.shape == (N,)
        assert init_psi.shape == (N,)
        assert self.v_arr.shape == (N,)

        self.hbar = hbar
        self.m = m
        self.t = t_init
        self.dt_ = None
        self.N = len(x)
        self.dx = self.x[1] - self.x[0]
        self.dk = 2 * np.pi / (self.N * self.dx)

        if k_min == None:
            self.k_min = -0.5 * self.N * self.dk
        else:
            self.k_min = k_min
        self.k = self.k_min + self.dk * np.arange(self.N)

        self.psi_of_x = init_psi
        self.k_fft()

        self.x_evolve_half = None
        self.x_evolve = None
        self.k_evolve = None

        self.psi_of_x_line = None
        self.psi_of_k_line = None
        self.v_arr_line = None

    def _set_psi_of_x(self, psi_of_x):
        self.psi_mod_x = (psi_of_x * np.exp(-1j * self.k[0] * self.x)
                          * self.dx / np.sqrt(2 * np.pi))

    def _psi_of_x(self):
        return (self.psi_mod_x * np.exp(1j * self.k[0] * self.x)
                * np.sqrt(2 * np.pi) / self.dx)

    def _set_psi_of_k(self, psi_of_k):
        self.psi_mod_k = psi_of_k * np.exp(1j * self.x[0]
                                        * self.dk * np.arange(self.N))

    def _get_psi_of_k(self):
        return self.psi_mod_k * np.exp(-1j * self.x[0] * 
                                        self.dk * np.arange(self.N))
    
    def _get_dt(self):
        return self.dt_

    def _set_dt(self, dt):
        if dt != self.dt_:
            self.dt_ = dt
            self.x_evolve_half = np.exp(-0.5 * 1j * self.v_arr
                                         / self.hbar * dt )
            self.x_evolve = self.x_evolve_half * self.x_evolve_half
            self.k_evolve = np.exp(-0.5 * 1j * self.hbar /
                                    self.m * (self.k * self.k) * dt)
    
    psi_of_x = property(_psi_of_x, _set_psi_of_x)
    psi_of_k = property(_get_psi_of_k, _set_psi_of_k)
    dt = property(_get_dt, _set_dt)

    def k_fft(self):
        self.psi_mod_k = fft(self.psi_mod_x)

    def x_fft(self):
        self.psi_mod_x = ifft(self.psi_mod_k)

    def time_step(self, dt, Nsteps = 1):
        """
        Perform a series of time-steps via the time-dependent
        Schrodinger Equation.

        Parameters
        ----------
        dt : float
            the small time interval over which to integrate
        Nsteps : float, optional
            the number of intervals to compute.  The total change
            in time at the end of this method will be dt * Nsteps.
            default is N = 1
        """
        self.dt = dt

        if Nsteps > 0:
            self.psi_mod_x *= self.x_evolve_half
    
        for i in xrange(Nsteps - 1):
            self.k_fft()
            self.psi_mod_k *= self.k_evolve
            self.x_fft()
            self.psi_mod_x *= self.x_evolve
            
        self.k_fft()

        self.t += dt * Nsteps


def gauss_x(x, a, x0, k_min):
 
    return ((a * np.sqrt(np.pi)) ** (-0.5)
            * np.exp(-0.5 * ((x - x0) * 1. / a) ** 2 + 1j * x * k_min))

def gauss_k(k,a,x0,k_min):
 
    return ((a / np.sqrt(np.pi))**0.5
            * np.exp(-0.5 * (a * (k - k_min)) ** 2 - 1j * (k - k_min) * x0))


def theta(x):
    """
    theta function :
      returns 0 if x<=0, and 1 if x>0
    """
    x = np.asarray(x)
    y = np.zeros(x.shape)
    #y = (x**2)/1000
    y = (x/5)**2
    return y

def square_barrier(x, width, height):
    return theta(x)/30

dt = 0.01
N_steps = 50
t_max = 120
frames = int(t_max / float(N_steps * dt))


N = 2 ** 11
dx = 0.1
x = dx * (np.arange(N) - 0.5 * N)

V0 = 1.5
L = hbar / np.sqrt(2 * m * V0)
a = 300 * L
x0 = -60 * L
v_arr = square_barrier(x, a, V0)
v_arr[x < -98] = 1E6
v_arr[x > 98] = 1E6

p0 = np.sqrt(2 * m * 0.2 * V0)
dp2 = p0 * p0 * 1./80
d = hbar / np.sqrt(2 * dp2)

k_min = p0 / hbar
v0 = p0 / m
init_psi = gauss_x(x, d, x0, k_min)

S = Schrodinger(x=x,
                init_psi=init_psi,
                v_arr=v_arr,
                hbar=1,
                m=1,
                k_min=-28)

fig = pl.figure()

xlim = (-100, 100)
klim = (-5, 5)

ymin = 0
ymax = V0
ax1 = fig.add_subplot(121, xlim=xlim,
                      ylim=(ymin - 0.2 * (ymax - ymin),
                            ymax + 0.2 * (ymax - ymin)))
psi_of_x_line, = ax1.plot([], [], c='r', label=r'$|\psi(x)|$')
v_arr_line, = ax1.plot([], [], c='k', label=r'$V(x)$')
center_line = ax1.axvline(0, c='k', ls=':',
                          label = r"$x_0 + v_0t$")

title = ax1.set_title("")
ax1.set_ylabel('psi(x)')
ax1.set_xlabel('position')

ymin = abs(S.psi_of_k).min()
ymax = abs(S.psi_of_k).max()
ax2 = fig.add_subplot(122, xlim=klim,
                      ylim=(ymin - 0.2 * (ymax - ymin),
                            ymax + 0.2 * (ymax - ymin)))
psi_of_k_line, = ax2.plot([], [], c='r', label=r'$|\psi(k)|$')


ax2.set_ylabel('psi(k)')
ax2.set_xlabel('momentum')

def init():
    psi_of_x_line.set_data([], [])
    v_arr_line.set_data([], [])
    center_line.set_data([], [])

    psi_of_k_line.set_data([], [])
    title.set_text("")
    return (psi_of_x_line,psi_of_k_line, title)

def animate(i):
    S.time_step(dt, N_steps)
    psi_of_x_line.set_data(S.x, 4 * abs(S.psi_of_x))
    v_arr_line.set_data(S.x, S.v_arr)
    #center_line.set_data(2 * [x0 + S.t * p0 / m], [0, 1])

    psi_of_k_line.set_data(S.k, abs(S.psi_of_k))
    title.set_text("t = %.2f" % S.t)
    return (psi_of_x_line,psi_of_k_line, title)

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=30, blit=False)
