#!/usr/bin/env python3
"""
Random Walk -> Diffusion Equation lecture demo (Tk GUI).

Two tabs:

1. "Random walk intuition"
   Simulate many discrete random walkers. Watch trajectories fan out and the
   histogram of positions widen as the number of steps increases. Slider lets
   you scrub through time; a checkbox overlays a Gaussian fit so students see
   the shape emerging empirically, before any equation is written down.

2. "Diffusion equation vs. random walk"
   The same discrete walk (now parameterized by physical step size dx and
   time increment dt) is compared against a finite-difference solution of the
   drift-diffusion PDE

       dP/dt = -v dP/dx + D d^2P/dx^2

   where v and D are the drift and diffusion coefficients implied by the
   discrete jump process (dx, dt, p). This is the object you get by
   Taylor-expanding the master equation P(x,t+dt) = p P(x-dx,t) + (1-p) P(x+dx,t).
   Moving the p-slider away from 0.5 adds a drift term, useful as an optional
   extension after the symmetric case is understood.

Run with:
    python3 random_walk_diffusion_gui.py

Requires: numpy, matplotlib, tkinter (standard library on most Python
installs; on some Linux distros install via your package manager, e.g.
`sudo apt install python3-tk`). Everything is in this single file.
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ----------------------------------------------------------------------------
# Core numerical routines
#
# Convention: a walker takes steps of size +/- dx, with P(+dx)=p, P(-dx)=1-p,
# once every dt (time units). After n steps, real time is t = n * dt.
#
# Per-step statistics:
#   mean displacement  mu  = dx * (2p - 1)
#   variance            var = dx^2 * (1 - (2p-1)^2) = 4 * dx^2 * p * (1-p)
#
# Continuum (drift-diffusion) limit as dx, dt -> 0 with mu/dt, var/dt fixed:
#   drift v = mu / dt    = dx * (2p-1) / dt
#   D       = var / (2dt) = 2 * dx^2 * p * (1-p) / dt
#
# so the master equation P(x,t+dt) = p P(x-dx,t) + (1-p) P(x+dx,t) Taylor-
# expands to dP/dt = -v dP/dx + D d^2P/dx^2.
# ----------------------------------------------------------------------------

def simulate_random_walk(n_walkers, n_steps, dx, dt, p, seed=None):
    """
    Simulate n_walkers independent discrete random walks.

    Returns
    -------
    positions : ndarray, shape (n_walkers, n_steps + 1); positions[:,0] = 0.
    times     : ndarray, shape (n_steps + 1,); times[k] = k * dt.
    """
    rng = np.random.default_rng(seed)
    draws = rng.random((n_walkers, n_steps))
    steps = np.where(draws < p, dx, -dx)
    positions = np.zeros((n_walkers, n_steps + 1))
    positions[:, 1:] = np.cumsum(steps, axis=1)
    times = np.arange(n_steps + 1) * dt
    return positions, times


def drift_and_diffusion(dx, dt, p):
    """Return (v, D) implied by the discrete jump process."""
    v = dx * (2 * p - 1) / dt
    D = 2 * dx**2 * p * (1 - p) / dt
    return v, D


def analytic_gaussian(x, t, D, v=0.0, t0=1e-12):
    """
    Closed-form solution of dP/dt = -v dP/dx + D d^2P/dx^2
    for a point source at x=0, t=0 (t0 avoids division by zero at t=0).
    """
    t_eff = max(t, t0)
    return (1.0 / np.sqrt(4 * np.pi * D * t_eff)) * np.exp(
        -((x - v * t_eff) ** 2) / (4 * D * t_eff)
    )


def solve_diffusion_fd(D, v, t_final, x_min, x_max, n_grid=401, cfl=0.4, sigma0_cells=1.0):
    """
    Explicit (FTCS) finite-difference solution of
        dP/dt = -v dP/dx + D d^2P/dx^2
    on [x_min, x_max] with a narrow Gaussian initial condition centered at 0
    (a numerical stand-in for a delta function, width = sigma0_cells * dx_grid).

    Returns
    -------
    x_grid : ndarray, shape (n_grid,)
    P      : ndarray, shape (n_grid,)  -- solution at t_final
    """
    x_grid = np.linspace(x_min, x_max, n_grid)
    dxg = x_grid[1] - x_grid[0]

    sigma0 = max(sigma0_cells * dxg, 1e-9)
    P = np.exp(-x_grid**2 / (2 * sigma0**2))
    P /= np.trapezoid(P, x_grid)  # normalize to a unit "probability mass"

    if t_final <= 0 or D <= 0:
        return x_grid, P

    # Stability: explicit diffusion needs D*dt/dx^2 <= 0.5; advection (CFL)
    # needs |v|*dt/dx <= 1. Pick dt from whichever constraint is tighter.
    dt_diff = cfl * dxg**2 / D
    dt_adv = 0.5 * dxg / abs(v) if abs(v) > 1e-12 else np.inf
    dt_pde = min(dt_diff, dt_adv)

    n_sub = max(1, int(np.ceil(t_final / dt_pde)))
    dt_pde = t_final / n_sub

    r = D * dt_pde / dxg**2
    c = v * dt_pde / (2 * dxg)

    for _ in range(n_sub):
        P_left = np.roll(P, 1)
        P_right = np.roll(P, -1)
        P_new = P + r * (P_right - 2 * P + P_left) - c * (P_right - P_left)
        # absorbing-ish boundaries: hold edges at ~0
        P_new[0] = 0.0
        P_new[-1] = 0.0
        P = P_new

    return x_grid, P


def lattice_bins(data, dx, target_bins=25):
    """
    Build histogram bin edges that are wide enough to look smooth and are
    aligned to multiples of dx.

    A discrete random walk with step size dx only visits lattice sites
    ..., -2dx, -dx, 0, dx, 2dx, ... and, after n steps, only sites of one
    parity (all even or all odd multiples of dx) are reachable. Bins that are
    narrower than 2*dx (or not aligned to the lattice) produce a jagged
    "comb" pattern of alternating full/empty bins. Rounding the bin width up
    to a multiple of dx (at least 2*dx) and aligning edges to the lattice
    avoids that artifact and gives a clean-looking histogram.
    """
    span = max(np.abs(data).max(), dx) * 1.15 if len(data) else dx * 1.15
    raw_width = (2 * span) / max(target_bins, 1)
    n_dx = max(2, round(raw_width / dx))
    if n_dx % 2 != 0:
        n_dx += 1  # keep bin width an even multiple of dx so it always spans
                    # both site parities, regardless of step count
    width = n_dx * dx
    n_half = int(np.ceil(span / width)) + 1
    edges = np.arange(-n_half, n_half + 1) * width
    return edges


# ----------------------------------------------------------------------------
# Reusable labeled-slider widget
# ----------------------------------------------------------------------------
class LabeledSlider(ttk.Frame):
    def __init__(self, parent, label, from_, to, initial, resolution=1,
                 is_int=False, command=None, width=220):
        super().__init__(parent)
        self.is_int = is_int
        self.var = tk.DoubleVar(value=initial)
        self._command = command

        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text=label).pack(side="left")
        self.value_label = ttk.Label(top, text=self._fmt(initial), width=8, anchor="e")
        self.value_label.pack(side="right")

        self.scale = ttk.Scale(
            self, from_=from_, to=to, orient="horizontal",
            variable=self.var, command=self._on_change, length=width
        )
        self.scale.pack(fill="x")
        self.resolution = resolution

    def _fmt(self, v):
        return f"{int(round(v))}" if self.is_int else f"{v:.3g}"

    def _on_change(self, _evt=None):
        v = self.var.get()
        if self.resolution:
            v = round(v / self.resolution) * self.resolution
        if self.is_int:
            v = int(round(v))
        self.value_label.config(text=self._fmt(v))
        if self._command:
            self._command(v)

    def get(self):
        v = self.var.get()
        if self.resolution:
            v = round(v / self.resolution) * self.resolution
        return int(round(v)) if self.is_int else v

    def set(self, v):
        self.var.set(v)
        self.value_label.config(text=self._fmt(v))


# ----------------------------------------------------------------------------
# Tab 1: Random walk intuition
# ----------------------------------------------------------------------------
class RandomWalkTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.positions = None
        self.times = None
        self._build_ui()
        self.resimulate()

    def _build_ui(self):
        controls = ttk.Frame(self, padding=10)
        controls.pack(side="left", fill="y")

        ttk.Label(controls, text="Random walk intuition", font=("", 12, "bold")).pack(
            anchor="w", pady=(0, 10)
        )

        self.n_walkers = LabeledSlider(controls, "Number of walkers", 1, 10000, 200,
                                       resolution=1, is_int=True)
        self.n_walkers.pack(fill="x", pady=4)

        self.n_steps = LabeledSlider(controls, "Number of steps", 10, 2000, 500,
                                     resolution=1, is_int=True)
        self.n_steps.pack(fill="x", pady=4)

        self.p_right = LabeledSlider(controls, "P(step right)", 0.0, 1.0, 0.5,
                                     resolution=0.01)
        self.p_right.pack(fill="x", pady=4)

        self.n_shown = LabeledSlider(controls, "Trajectories drawn", 1, 200, 30,
                                     resolution=1, is_int=True)
        self.n_shown.pack(fill="x", pady=4)

        self.n_bins = LabeledSlider(controls, "Histogram bins (approx.)", 5, 60, 20,
                                    resolution=1, is_int=True, command=lambda _v: self.redraw())
        self.n_bins.pack(fill="x", pady=4)

        ttk.Button(controls, text="Re-simulate (new random draws)",
                   command=self.resimulate).pack(fill="x", pady=(10, 4))

        ttk.Separator(controls, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(controls, text="Scrub through time:").pack(anchor="w")
        self.t_slider = LabeledSlider(controls, "Step shown", 0, 500, 500,
                                      resolution=1, is_int=True, command=self._on_time_change)
        self.t_slider.pack(fill="x", pady=4)

        self.show_gaussian = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Overlay Gaussian fit",
                        variable=self.show_gaussian,
                        command=self.redraw).pack(anchor="w", pady=(6, 0))

        self.play_var = tk.BooleanVar(value=False)
        self.play_btn = ttk.Button(controls, text="Play", command=self.toggle_play)
        self.play_btn.pack(fill="x", pady=(10, 4))

        self.stats_label = ttk.Label(controls, text="", justify="left")
        self.stats_label.pack(anchor="w", pady=(10, 0))

        # Plot area
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side="right", fill="both", expand=True)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax_traj = self.fig.add_subplot(211)
        self.ax_hist = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def resimulate(self):
        n_walkers = self.n_walkers.get()
        n_steps = self.n_steps.get()
        p = self.p_right.get()
        self.positions, self.times = simulate_random_walk(
            n_walkers, n_steps, dx=1.0, dt=1.0, p=p, seed=None
        )
        self.t_slider.scale.config(to=n_steps)
        self.t_slider.set(n_steps)
        self.redraw()

    def _on_time_change(self, _v):
        self.redraw()

    def toggle_play(self):
        self.play_var.set(not self.play_var.get())
        if self.play_var.get():
            self.play_btn.config(text="Pause")
            self._animate_step()
        else:
            self.play_btn.config(text="Play")

    def _animate_step(self):
        if not self.play_var.get():
            return
        cur = self.t_slider.get()
        n_steps = self.n_steps.get()
        nxt = cur + max(1, n_steps // 100)
        if nxt > n_steps:
            nxt = 0
        self.t_slider.set(nxt)
        self.redraw()
        self.after(40, self._animate_step)

    def redraw(self):
        if self.positions is None:
            return
        t_idx = self.t_slider.get()
        n_shown = min(self.n_shown.get(), self.positions.shape[0])

        self.ax_traj.clear()
        self.ax_hist.clear()

        # Trajectories up to t_idx
        for i in range(n_shown):
            self.ax_traj.plot(self.times[: t_idx + 1], self.positions[i, : t_idx + 1],
                              linewidth=0.7, alpha=0.7)
        self.ax_traj.axvline(self.times[t_idx], color="k", linewidth=0.8, linestyle="--")
        self.ax_traj.set_title(f"{n_shown} sample trajectories (step {t_idx})")
        self.ax_traj.set_xlabel("step (time)")
        self.ax_traj.set_ylabel("position")

        final = self.positions[:, t_idx]
        bins = lattice_bins(final, dx=1.0, target_bins=self.n_bins.get())
        self.ax_hist.hist(final, bins=bins, density=True, alpha=0.7,
                          color="tab:blue", label="walker histogram")

        if self.show_gaussian.get() and t_idx > 0:
            p = self.p_right.get()
            v, D = drift_and_diffusion(1.0, 1.0, p)
            t = self.times[t_idx]
            xs = np.linspace(bins[0], bins[-1], 300)
            self.ax_hist.plot(xs, analytic_gaussian(xs, t, D, v),
                              color="tab:red", linewidth=2, label="Gaussian fit")
            self.ax_hist.legend(loc="upper right", fontsize=8)

        self.ax_hist.set_title("Histogram of positions at this step")
        self.ax_hist.set_xlabel("position")
        self.ax_hist.set_ylabel("density")

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw_idle()

        mean = final.mean()
        var = final.var()
        self.stats_label.config(
            text=(f"step = {t_idx}\n"
                  f"mean position = {mean:.2f}\n"
                  f"variance      = {var:.2f}\n"
                  f"(watch: variance grows ~linearly in step count)")
        )


# ----------------------------------------------------------------------------
# Tab 2: Diffusion equation vs. random walk histogram
# ----------------------------------------------------------------------------
class DiffusionTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.positions = None
        self.times = None
        self._build_ui()
        self.resimulate()

    def _build_ui(self):
        controls = ttk.Frame(self, padding=10)
        controls.pack(side="left", fill="y")

        ttk.Label(controls, text="Diffusion equation vs. random walk",
                 font=("", 12, "bold")).pack(anchor="w", pady=(0, 10))

        self.n_walkers = LabeledSlider(controls, "Number of walkers", 100, 20000, 5000,
                                       resolution=100, is_int=True)
        self.n_walkers.pack(fill="x", pady=4)

        self.n_steps = LabeledSlider(controls, "Number of steps", 10, 2000, 800,
                                     resolution=1, is_int=True)
        self.n_steps.pack(fill="x", pady=4)

        self.dx = LabeledSlider(controls, "Step size  Δx", 0.1, 3.0, 1.0,
                                resolution=0.1)
        self.dx.pack(fill="x", pady=4)

        self.dt = LabeledSlider(controls, "Time increment  Δt", 0.1, 3.0, 1.0,
                                resolution=0.1)
        self.dt.pack(fill="x", pady=4)

        self.p_right = LabeledSlider(controls, "P(step right)  [0.5 = pure diffusion]",
                                     0.0, 1.0, 0.5, resolution=0.01)
        self.p_right.pack(fill="x", pady=4)

        self.n_bins = LabeledSlider(controls, "Histogram bins (approx.)", 5, 60, 20,
                                    resolution=1, is_int=True, command=lambda _v: self.redraw())
        self.n_bins.pack(fill="x", pady=4)

        ttk.Button(controls, text="Re-simulate (new random draws)",
                   command=self.resimulate).pack(fill="x", pady=(10, 4))

        ttk.Separator(controls, orient="horizontal").pack(fill="x", pady=10)

        self.t_slider = LabeledSlider(controls, "Step shown", 1, 800, 800,
                                      resolution=1, is_int=True, command=self._on_time_change)
        self.t_slider.pack(fill="x", pady=4)

        self.play_var = tk.BooleanVar(value=False)
        self.play_btn = ttk.Button(controls, text="Play", command=self.toggle_play)
        self.play_btn.pack(fill="x", pady=(10, 4))

        self.show_analytic = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Also show closed-form Gaussian",
                        variable=self.show_analytic,
                        command=self.redraw).pack(anchor="w", pady=(6, 0))

        self.info_label = ttk.Label(controls, text="", justify="left", wraplength=240)
        self.info_label.pack(anchor="w", pady=(10, 0))

        plot_frame = ttk.Frame(self)
        plot_frame.pack(side="right", fill="both", expand=True)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def resimulate(self):
        n_walkers = self.n_walkers.get()
        n_steps = self.n_steps.get()
        dx = self.dx.get()
        dt = self.dt.get()
        p = self.p_right.get()
        self.positions, self.times = simulate_random_walk(
            n_walkers, n_steps, dx=dx, dt=dt, p=p, seed=None
        )
        self.t_slider.scale.config(to=n_steps)
        self.t_slider.set(n_steps)
        self.redraw()

    def _on_time_change(self, _v):
        self.redraw()

    def toggle_play(self):
        self.play_var.set(not self.play_var.get())
        if self.play_var.get():
            self.play_btn.config(text="Pause")
            self._animate_step()
        else:
            self.play_btn.config(text="Play")

    def _animate_step(self):
        if not self.play_var.get():
            return
        cur = self.t_slider.get()
        n_steps = self.n_steps.get()
        nxt = cur + max(1, n_steps // 100)
        if nxt > n_steps:
            nxt = 1
        self.t_slider.set(nxt)
        self.redraw()
        self.after(40, self._animate_step)

    def redraw(self):
        if self.positions is None:
            return
        t_idx = self.t_slider.get()
        dx = self.dx.get()
        dt = self.dt.get()
        p = self.p_right.get()
        v, D = drift_and_diffusion(dx, dt, p)
        t = self.times[t_idx]

        final = self.positions[:, t_idx]

        self.ax.clear()

        span = max(np.abs(final).max(), dx) * 1.15
        bins = lattice_bins(final, dx=dx, target_bins=self.n_bins.get())
        self.ax.hist(final, bins=bins, density=True, alpha=0.6, color="tab:blue",
                    label=f"random walk histogram (N={len(final)})")

        x_grid, P_fd = solve_diffusion_fd(D, v, t, x_min=-span, x_max=span, n_grid=401)
        self.ax.plot(x_grid, P_fd, color="tab:green", linewidth=2.5,
                    label="diffusion PDE (finite-difference)")

        if self.show_analytic.get() and t > 0:
            xs = np.linspace(-span, span, 300)
            self.ax.plot(xs, analytic_gaussian(xs, t, D, v), color="tab:red",
                        linewidth=1.5, linestyle="--", label="closed-form Gaussian")

        self.ax.set_title(f"t = {t:.2f}   (step {t_idx} of {self.n_steps.get()})")
        self.ax.set_xlabel("position x")
        self.ax.set_ylabel("probability density")
        self.ax.legend(loc="upper right", fontsize=8)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw_idle()

        self.info_label.config(
            text=(f"Implied drift  v = {v:.3f}\n"
                  f"Implied diffusion coeff.  D = {D:.3f}\n"
                  f"(D = 2Δx²p(1-p)/Δt,  v = Δx(2p-1)/Δt)\n\n"
                  f"empirical mean = {final.mean():.3f} (theory: {v*t:.3f})\n"
                  f"empirical var  = {final.var():.3f} (theory: {2*D*t:.3f})")
        )


# ----------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Random Walks -> Diffusion Equation")
        self.geometry("1150x720")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        tab1 = RandomWalkTab(notebook)
        tab2 = DiffusionTab(notebook)

        notebook.add(tab1, text="1. Random walk intuition")
        notebook.add(tab2, text="2. Diffusion eq. vs. histogram")


if __name__ == "__main__":
    app = App()
    app.mainloop()