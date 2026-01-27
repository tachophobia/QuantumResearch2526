import scipy.sparse as spmat
import scipy.sparse.linalg as spla
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def time_evolve(domain, psi_0, Vx, max_t, it=1000, fix_norm=False):
    dx = np.abs(domain[1] - domain[0])
    dt = max_t / it
    if dt > dx**2:
        print(f"Warning: CFL stability may be violated. dx = {dx}, dt = {dt}.")
    N = len(domain)

    psi_trotter = psi_0.copy()
    psi_crank = psi_0.copy()
    psi_avg = psi_0.copy()
    
    # crank setup
    main = 2.0*np.ones(N)/dx**2 + Vx
    off  = -1.0*np.ones(N-1)/dx**2
    H = spmat.diags([off, main, off], [-1,0,1], format='csc')
    I = spmat.identity(N, format="csc")
    A = I + 1j*(dt/2)*H
    B = I - 1j*(dt/2)*H
    A_factor = spla.factorized(A)

    # trotter setup
    k = np.fft.fftfreq(N, d=dx) * 2*np.pi

    results = {
        't': [],'x': [], 'p': [], 'T': [], 'V': [],
        'psi_avg': [],
        'err_psi': [],
        'err_norm': [],
        'err_norm_crank': [],
        'err_norm_trot': [],
    }

    for j in tqdm(range(it+1)):
        results['t'].append(j*dt)
        results['psi_avg'].append(psi_avg.copy())
        
        # expectation values
        dpsi_dx = np.gradient(psi_avg, dx)
        d2psi_dx2 = np.gradient(dpsi_dx, dx)
        results['x'].append(np.trapz(np.conjugate(psi_avg)*domain*psi_avg, x=domain).real)
        results['p'].append(np.trapz(np.conjugate(psi_avg) * -1j * dpsi_dx, x=domain).real)
        results['T'].append(np.trapz(np.conjugate(psi_avg) * (-d2psi_dx2), x=domain).real)
        results['V'].append(np.trapz(np.conjugate(psi_avg) * Vx * psi_avg, x=domain).real)

        # error tracking
        results['err_psi'].append(np.sqrt(np.trapz(np.abs(psi_crank - psi_trotter)**2, x=domain)))
        results['err_norm'].append(np.sqrt(np.trapz(np.abs(psi_avg)**2, x=domain)) - 1)
        norm_crank = np.trapz(np.abs(psi_crank)**2, x=domain)
        results['err_norm_crank'].append(np.sqrt(norm_crank) - 1)
        norm_trot = np.trapz(np.abs(psi_trotter)**2, x=domain)
        results['err_norm_trot'].append(np.sqrt(norm_trot) - 1)

        # crank update
        rhs = B.dot(psi_crank)
        psi_crank = A_factor(rhs)
        if fix_norm:
            psi_crank /= np.sqrt(norm_crank)

        # trotter update
        psi_trotter = np.exp(-1j * Vx * (dt/2)) * psi_trotter
        psi_trotter = np.fft.ifft( np.exp(-1j * (k**2) * dt) * np.fft.fft(psi_trotter) )
        psi_trotter = np.exp(-1j * Vx * (dt/2)) * psi_trotter
        if fix_norm:
            psi_trotter /= np.sqrt(norm_trot)

        # avg update
        psi_avg = (psi_crank + psi_trotter) / 2

    return results


def animate(
    domain,
    psis,
    max_t,
    figsize=(6, 4),
    ylabel=r"$|\psi(x,t)|^2$",
    interval=50,
    filename="psi.gif",
    dpi=100
):
    if psis.ndim != 2:
        raise ValueError("psis must have shape (Nt, Nx)")

    Nt, Nx = psis.shape
    t = np.linspace(0, max_t, Nt)
    if len(t) != Nt:
        raise ValueError("Length of t must match number of frames")

    data = psis.real

    y_min = data.min()
    y_max = data.max()

    fig, ax = plt.subplots(figsize=figsize)
    line, = ax.plot(domain, data[0], lw=2)

    ax.set_xlim(domain.min(), domain.max())
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)

    time_text = ax.text(
        0.02, 0.95, "",
        transform=ax.transAxes,
        ha="left",
        va="top"
    )

    def init():
        line.set_ydata(data[0])
        time_text.set_text(f"t = {t[0]:.4f}")
        return line, time_text

    def update(frame):
        line.set_ydata(data[frame])
        time_text.set_text(f"t = {t[frame]:.4f}")
        return line, time_text

    anim = FuncAnimation(
        fig,
        update,
        frames=Nt,
        init_func=init,
        interval=interval,
        blit=True
    )

    writer = PillowWriter(fps=1000 // interval)
    anim.save(filename, writer=writer, dpi=dpi)

