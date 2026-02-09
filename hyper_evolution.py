import scipy.sparse as spmat
import scipy.sparse.linalg as spla
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from hyper_eigenstates import change_basis_hyperbolic
from matplotlib.collections import LineCollection


def time_evolve(domain, psi_0, Vx, max_t, it=1000, fix_norm=False, hyperbolic=False):
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
        'unbound_psi': []
    }

    if hyperbolic:
        overlap = change_basis_hyperbolic(domain, psi_0, 3)

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
        if hyperbolic:
            bound = sum([np.trapz(np.conjugate(psi_n) * psi_crank, x=domain) * psi_n for _, _, psi_n in overlap])
            results['unbound_psi'].append(psi_crank - bound)

        if fix_norm:
            psi_crank /= np.sqrt(norm_crank)

        # trotter update
        psi_trotter = np.exp(-1j * Vx * (dt/2)) * psi_trotter
        psi_trotter = np.fft.ifft( np.exp(-1j * (k**2) * dt) * np.fft.fft(psi_trotter) )
        psi_trotter = np.exp(-1j * Vx * (dt/2)) * psi_trotter
        if hyperbolic:
            pass
            # psi_trotter = sum([np.trapz(np.conjugate(psi_n) * psi_trotter, x=domain) * psi_n for _, _, psi_n in overlap])
        if fix_norm:
            psi_trotter /= np.sqrt(norm_trot)

        # avg update
        psi_avg = (psi_crank + psi_trotter) / 2

    return results


from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

def animate(
    domain,
    psis,
    max_t,
    unbound_psi=None,
    figsize=(6, 4),
    ylabel=r"$|\psi(x,t)|^2$",
    interval=50,
    filename="psi.gif",
    dpi=100,
    x_condense_start=None
):
    if psis.ndim != 2:
        raise ValueError("psis must have shape (Nt, Nx)")

    Nt, Nx = psis.shape
    t = np.linspace(0, max_t, Nt)
    data = np.abs(psis)**2

    fig, ax = plt.subplots(figsize=figsize)

    # --------------------------------------------------
    # Axis scaling MUST be done explicitly for collections
    # --------------------------------------------------
    ax.set_xlim(domain.min(), domain.max())
    ax.set_ylim(data.min(), data.max())

    if x_condense_start is not None:
        ax.set_xscale("symlog", linthresh=x_condense_start)

    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)

    # --------------------------------------------------
    # Prepare percent-change data (spatially meaningful)
    # --------------------------------------------------
    if unbound_psi is not None:
        unbound_data = np.abs(unbound_psi)**2

        pct_change = np.zeros_like(unbound_data)
        pct_change[1:] = np.abs(unbound_data[1:] - unbound_data[:-1])

        # Normalize ONCE, globally (important!)
        norm = Normalize(
            vmin=0.0,
            vmax=pct_change.max() if pct_change.max() > 0 else 1.0
        )
    else:
        pct_change = None
        norm = None

    # --------------------------------------------------
    # Initial line geometry
    # --------------------------------------------------
    y0 = data[0]
    points = np.column_stack((domain, y0))
    segments = np.stack((points[:-1], points[1:]), axis=1)

    lc = LineCollection(
        segments,
        cmap="plasma",
        norm=norm,
        linewidth=2,
        animated=True
    )

    if pct_change is not None:
        lc.set_array(pct_change[0, :-1])

    ax.add_collection(lc)

    time_text = ax.text(
        0.02, 0.95, "",
        transform=ax.transAxes,
        ha="left", va="top",
        animated=True
    )

    # --------------------------------------------------
    # Init
    # --------------------------------------------------
    def init():
        lc.set_segments(segments)
        if pct_change is not None:
            lc.set_array(pct_change[0, :-1])
        time_text.set_text(f"t = {t[0]:.4f}")
        return lc, time_text

    # --------------------------------------------------
    # Update
    # --------------------------------------------------
    def update(frame):
        y = data[frame]
        points[:, 1] = y

        segs = np.stack((points[:-1], points[1:]), axis=1)
        lc.set_segments(segs)

        if pct_change is not None:
            lc.set_array(pct_change[frame, :-1])

        time_text.set_text(f"t = {t[frame]:.4f}")
        return lc, time_text

    anim = FuncAnimation(
        fig,
        update,
        frames=Nt,
        init_func=init,
        interval=interval,
        blit=True
    )

    writer = PillowWriter(fps=max(1, 1000 // interval))
    anim.save(filename, writer=writer, dpi=dpi)
