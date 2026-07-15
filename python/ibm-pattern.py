"""
1D individual-based birth-death model with long-range dispersal and
competition, plus its mean-field (density-field) PDE approximation.

Model (Surendran, Pinto-Ramos, Menezes & Martinez-Garcia, 2025,
"Spatial moment dynamics and biomass density equations provide
complementary, yet limited, descriptions of pattern formation in
individual-based simulations", Physica D 477, 134703):

  * Each organism at x_i gives birth at rate  b * phi_D(x - x_i)
    (Gaussian dispersal kernel, std s), and dies at rate
    d + g * sum_j phi_C(x_i - x_j)  (competition kernel).
  * phi_C is a top-hat kernel of half-width r_c (Eq. 2.3 in the paper).
    This is the kernel choice shown in the paper to produce *periodic*
    (regularly spaced) aggregation patterns (Section 3.2), which is why
    it -- rather than a Gaussian competition kernel -- is used here.

The lattice/master-equation dynamics are simulated with a
tau-leaping (Poisson) approximation, which is the discretised,
simulate-able version of the spatial Master equation (Eq. 2.15-2.19).

The mean-field density equation shown alongside it is obtained exactly
the way it is in the paper (Eq. 2.20-2.21): take the expectation of the
master equation and neglect the spatial covariance
(Cov(x,x') = 0, Eq. 2.22-2.23). That gives a closed, deterministic
equation on the same lattice, which is integrated forward with a simple
Euler step. Both the IBM and the PDE start from the *same* initial
lattice configuration so students can watch the stochastic realization
and the mean-field prediction diverge/converge as a pattern forms.

Run:
    python ibm_pattern_gui.py

Requires: numpy, matplotlib (uses the TkAgg backend; falls back to the
default backend if TkAgg is unavailable).
"""

import numpy as np
from collections import deque

import matplotlib
try:
    import tkinter  # noqa: F401  (probe availability before switching backend)
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button

# --------------------------------------------------------------------
# Domain
# --------------------------------------------------------------------
L = 20.0                       # periodic domain length
M = 200                        # number of lattice cells
dx = L / M
x = -L / 2 + dx * (np.arange(M) + 0.5)

# Minimal-image circular distance represented at FFT index k
_k = np.arange(M)
_shift = np.where(_k <= M // 2, _k, _k - M)
r_fft = _shift * dx

# --------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------
def gaussian_kernel(s):
    """Dispersal kernel phi_D: Gaussian with std s, normalized on the
    periodic lattice so that sum(kernel)*dx = 1."""
    phi = np.exp(-r_fft ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)
    phi /= phi.sum() * dx
    return phi


def tophat_kernel(rc):
    """Competition kernel phi_C: top-hat of half-width rc (Eq. 2.3),
    normalized on the periodic lattice."""
    phi = (np.abs(r_fft) <= rc).astype(float)
    s = phi.sum() * dx
    if s > 0:
        phi /= s
    return phi


def conv(N, kernel_fft):
    """Circular convolution (kernel * N), i.e. the discretised integral
    int phi(x - x') N(x') dx' on the periodic lattice."""
    return np.real(np.fft.ifft(np.fft.fft(N) * kernel_fft)) * dx


# --------------------------------------------------------------------
# Model parameters (defaults reproduce the periodic pattern in Fig. 2
# of the paper: s = 0.15, r_c = 2.2, b = 3.05, d = 0.305, g = 1)
# --------------------------------------------------------------------
params = dict(b=3.05, d=0.305, g=1.0, s=0.15, rc=2.2)

dt = 0.01          # tau-leap / Euler time step
DEFAULT_SUBSTEPS = 4   # simulation steps performed per animation frame (adjustable live)

rng = np.random.default_rng(0)
state = {}


def reinit(_event=None):
    """(Re)initialize both the IBM lattice and the PDE field from the
    same random draw around the homogeneous mean-field equilibrium."""
    nh = max(params['b'] - params['d'], 0.0) / params['g']   # homogeneous density
    N0 = rng.poisson(max(nh * dx, 1e-9), size=M).astype(float)
    state['N_ibm'] = N0.copy()
    state['N_pde'] = N0.copy()
    state['t'] = 0.0
    state['kernel_D_fft'] = np.fft.fft(gaussian_kernel(params['s']))
    state['kernel_C_fft'] = np.fft.fft(tophat_kernel(params['rc']))
    state['history'] = deque(maxlen=250)
    state['history'].append(N0.copy())
    state.setdefault('substeps', DEFAULT_SUBSTEPS)


reinit()


# --------------------------------------------------------------------
# Time stepping
# --------------------------------------------------------------------
def step_ibm(N, kD, kC):
    birth_rate = params['b'] * conv(N, kD)
    comp = conv(N, kC)
    death_rate = N * (params['d'] + params['g'] * comp)
    births = rng.poisson(np.clip(birth_rate * dt, 0, None))
    deaths = rng.poisson(np.clip(death_rate * dt, 0, None))
    deaths = np.minimum(deaths, N.astype(np.int64))
    N_new = N + births - deaths
    return np.clip(N_new, 0, None).astype(float)


def step_pde(N, kD, kC):
    birth_rate = params['b'] * conv(N, kD)
    comp = conv(N, kC)
    death_rate = N * (params['d'] + params['g'] * comp)
    N_new = N + dt * (birth_rate - death_rate)
    return np.clip(N_new, 0, None)


# --------------------------------------------------------------------
# Figure & widgets
# --------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
fig.suptitle("Spatial birth-death IBM vs. mean-field density equation (1D, top-hat competition)",
             fontsize=11)

ax_lattice = fig.add_axes([0.09, 0.56, 0.86, 0.34])
ax_kymo = fig.add_axes([0.09, 0.30, 0.86, 0.20])

bars = ax_lattice.bar(x, state['N_ibm'], width=dx * 0.9, color="#2e7d32", align="center",
                       label="IBM (individuals per cell)")
(pde_line,) = ax_lattice.plot(x, state['N_pde'], color="#d84315", lw=2.0,
                               label="Mean-field PDE", visible=False)
ax_lattice.set_xlim(-L / 2, L / 2)
ax_lattice.set_ylabel("individuals / cell")
ax_lattice.set_title("Lattice state")
ax_lattice.legend(loc="upper right", fontsize=8)
time_text = ax_lattice.text(0.01, 0.95, "", transform=ax_lattice.transAxes, va="top", fontsize=9)

im = ax_kymo.imshow(np.array(state['history']), aspect="auto", origin="lower",
                     extent=[-L / 2, L / 2, 0, 1], cmap="viridis")
ax_kymo.set_xlabel("space, x")
ax_kymo.set_ylabel("time")
ax_kymo.set_title("Space-time (kymograph) of the IBM density")

# --- sliders: single left column, stacked vertically ---
slider_axes = {
    'b':      fig.add_axes([0.24, 0.235, 0.35, 0.025]),
    'd':      fig.add_axes([0.24, 0.190, 0.35, 0.025]),
    's':      fig.add_axes([0.24, 0.145, 0.35, 0.025]),
    'rc':     fig.add_axes([0.24, 0.100, 0.35, 0.025]),
    'speed':  fig.add_axes([0.24, 0.055, 0.35, 0.025]),
}
sliders = {
    'b':  Slider(slider_axes['b'], 'Growth rate b', 0.5, 6.0, valinit=params['b']),
    'd':  Slider(slider_axes['d'], 'Death rate d', 0.05, 2.0, valinit=params['d']),
    's':  Slider(slider_axes['s'], 'Dispersal width s', 0.05, 1.5, valinit=params['s']),
    'rc': Slider(slider_axes['rc'], 'Competition (kernel) width r_c', 0.1, 5.0, valinit=params['rc']),
    'speed': Slider(slider_axes['speed'], 'Sim. speed (steps/frame)', 1, 25,
                     valinit=DEFAULT_SUBSTEPS, valstep=1),
}

# --- checkbox + buttons: right column, stacked vertically ---
check_ax = fig.add_axes([0.68, 0.19, 0.24, 0.06])
check = CheckButtons(check_ax, ["Show PDE solution"], [False])

pause_ax = fig.add_axes([0.68, 0.125, 0.24, 0.045])
pause_button = Button(pause_ax, "Pause")

reset_ax = fig.add_axes([0.68, 0.07, 0.24, 0.045])
reset_button = Button(reset_ax, "Reset lattice")

state['paused'] = False
state['show_pde'] = False


def on_slider_change(_val):
    for key in ('b', 'd', 's', 'rc'):
        params[key] = sliders[key].val
    reinit()


def on_speed_change(_val):
    # Only changes how many simulation steps happen per redraw; does not
    # touch the model parameters or reset the lattice.
    state['substeps'] = int(sliders['speed'].val)


for key in ('b', 'd', 's', 'rc'):
    sliders[key].on_changed(on_slider_change)
sliders['speed'].on_changed(on_speed_change)


def on_check(_label):
    state['show_pde'] = not state['show_pde']
    pde_line.set_visible(state['show_pde'])


check.on_clicked(on_check)


def on_pause(_event):
    state['paused'] = not state['paused']
    pause_button.label.set_text("Resume" if state['paused'] else "Pause")


pause_button.on_clicked(on_pause)


def on_reset(_event):
    reinit()


reset_button.on_clicked(on_reset)


# --------------------------------------------------------------------
# Animation
# --------------------------------------------------------------------
def update(_frame):
    n_steps = state['substeps']
    if not state['paused']:
        kD, kC = state['kernel_D_fft'], state['kernel_C_fft']
        for _ in range(n_steps):
            state['N_ibm'] = step_ibm(state['N_ibm'], kD, kC)
            state['N_pde'] = step_pde(state['N_pde'], kD, kC)
            state['t'] += dt
        state['history'].append(state['N_ibm'].copy())

    for rect, h in zip(bars, state['N_ibm']):
        rect.set_height(h)
    ymax = max(1.0, state['N_ibm'].max() * 1.2,
               state['N_pde'].max() * 1.2 if state['show_pde'] else 0.0)
    ax_lattice.set_ylim(0, ymax)

    if state['show_pde']:
        pde_line.set_ydata(state['N_pde'])

    hist_arr = np.array(state['history'])
    im.set_data(hist_arr)
    im.set_extent([-L / 2, L / 2, max(0.0, state['t'] - len(state['history']) * dt * n_steps), state['t']])
    im.set_clim(0, max(1.0, hist_arr.max()))

    time_text.set_text(
        f"t = {state['t']:.2f}   N_IBM = {state['N_ibm'].sum():.0f}"
        + (f"   N_PDE = {state['N_pde'].sum():.1f}" if state['show_pde'] else "")
    )
    return list(bars) + [pde_line, im, time_text]


from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)

if __name__ == "__main__":
    plt.show()