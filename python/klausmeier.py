"""
Interactive 2D simulator for the Klausmeier (1999) model of semiarid vegetation
patterning, "Regular and Irregular Patterns in Semiarid Vegetation", Science 284.

Model (nondimensionalized water W and plant biomass N, on domain X,Y with
periodic boundary conditions):

    dW/dt = a - W - W N^2 + v dW/dX + Dw * Laplacian(W)
    dN/dt = R * W N^2 - m N       + Dn * Laplacian(N)

    a  : rainfall / water input rate
    m  : plant mortality rate
    v  : water advection speed (downhill flow, in the -X direction)
    R  : plant growth yield per unit water consumed
    Dn : plant (biomass) diffusion coefficient  -- present in the original paper
    Dw : water diffusion coefficient -- set to 0 to recover the original model;
         included here as a second, adjustable diffusion since surface/subsurface
         water also spreads laterally in reality, and it lets you explore how the
         two diffusive processes compete with advection to shape the pattern.

Non-spatial (ODE) equilibria, used for the bifurcation diagram on the right:
    N = 0 (always exists, always locally stable)
    N != 0:   a = m (1 + N^2) / (R N)   <=>   m N^2 - a R N + m = 0
              N_pm = [ a R +/- sqrt((aR)^2 - 4 m^2) ] / (2 m)
    Fold (saddle-node) bifurcation at N* = 1, a_c = 2m/R: below a_c only the
    trivial (bare-soil) state exists; above a_c bare soil coexists with a stable
    vegetated state (upper branch) separated by an unstable state (lower branch).

GUI: built with Tkinter (ttk) + a matplotlib Figure embedded via
FigureCanvasTkAgg -- no matplotlib-native widgets/sliders are used, since those
were the source of the earlier layout/interaction problems.

    LEFT   : sliders for a, v, Dn, Dw, m, R + Reset/Pause buttons
    RIGHT-TOP    : live 2D field of vegetation biomass N(X,Y,t), periodic domain
    RIGHT-BOTTOM : bifurcation diagram of the non-spatial model (N* vs a) with a
                   marker at (a_current, mean N over the 2D domain), updated live

Run with:  python klausmeier_interactive.py
Requires:  numpy, matplotlib  (Tkinter ships with standard CPython on most
           platforms; on some Linux distros install it via e.g. `apt install
           python3-tk`).
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ----------------------------------------------------------------------------
# Domain / numerics
# ----------------------------------------------------------------------------
Nx, Ny = 128, 128
Lx, Ly = 200.0, 200.0          # domain size (same units as in the paper, ~meters)
dx, dy = Lx / Nx, Ly / Ny

# Fixed integration time step, chosen small enough to stay stable (CFL) across
# the whole slider range below (v up to 300, Dn up to 5, Dw up to 100).
DT = 0.0003
SUBSTEPS_PER_FRAME = 60         # PDE substeps integrated between redraws
FRAME_INTERVAL_MS = 50          # ~20 redraws/sec

# Hard safety caps on the fields. The reaction term W*N^2 has no built-in
# saturation, so pathological corners of parameter space (near-zero mortality
# combined with high rainfall/yield) can in principle grow without bound. These
# caps simply stop the simulation from ever producing NaN/Inf; they sit well
# above any ecologically meaningful value reached within the slider ranges.
N_CAP = 200.0
W_CAP = 50.0

# ----------------------------------------------------------------------------
# Default parameters (roughly the "grass" regime used in the paper's figures)
# ----------------------------------------------------------------------------
params = dict(a=2.0, v=150.0, Dn=1.0, Dw=10.0, m=0.45, R=1.0)


def laplacian(u):
    """5-point Laplacian with periodic boundary conditions."""
    return (
        np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0)
        + np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1)
        - 4.0 * u
    ) / (dx * dy)


def ddx_upwind(u, v):
    """First-order upwind derivative du/dX, periodic BC, for the term '+v du/dX'
    in dW/dt. Per the paper, water flows downhill in the -X direction at speed
    v>=0, so information travels from +X to -X: the stable upwind stencil looks
    toward +X (forward difference) when v>=0, and toward -X (backward
    difference) if v<0 (kept general in case a negative advection is dialed in).
    """
    if v >= 0:
        return (np.roll(u, -1, axis=0) - u) / dx
    else:
        return (u - np.roll(u, 1, axis=0)) / dx


def nontrivial_equilibrium(a, m, R):
    """Upper-branch homogeneous equilibrium (W+, N+), or None if it doesn't exist."""
    disc = (a * R) ** 2 - 4.0 * m ** 2
    if disc < 0:
        return None
    n_plus = (a * R + np.sqrt(disc)) / (2.0 * m)
    w_plus = m / (R * n_plus)
    return w_plus, n_plus


def init_fields():
    """Seed the field near the homogeneous equilibrium plus small noise, so
    patterns emerge from the linear instability rather than from a random mess."""
    eq = nontrivial_equilibrium(params["a"], params["m"], params["R"])
    if eq is None:
        w0, n0 = params["a"], 0.0
    else:
        w0, n0 = eq
    rng = np.random.default_rng(0)
    W = w0 * np.ones((Nx, Ny)) + 0.01 * rng.standard_normal((Nx, Ny))
    N = n0 * np.ones((Nx, Ny)) + 0.05 * max(n0, 0.1) * rng.standard_normal((Nx, Ny))
    W = np.clip(W, 0, None)
    N = np.clip(N, 0, None)
    return W, N


W, N = init_fields()


def step():
    """Advance (W, N) by SUBSTEPS_PER_FRAME explicit Euler steps."""
    global W, N
    a, v, Dn, Dw, m, R = (params[k] for k in ("a", "v", "Dn", "Dw", "m", "R"))
    for _ in range(SUBSTEPS_PER_FRAME):
        uptake = W * N ** 2
        dWdt = a - W - uptake + v * ddx_upwind(W, v) + Dw * laplacian(W)
        dNdt = R * uptake - m * N + Dn * laplacian(N)
        W = np.clip(W + DT * dWdt, 0, W_CAP)
        N = np.clip(N + DT * dNdt, 0, N_CAP)


def bifurcation_curves(a_axis, m, R):
    aR = a_axis * R
    disc = aR ** 2 - 4.0 * m ** 2
    valid = disc >= 0
    n_plus = np.full_like(a_axis, np.nan)
    n_minus = np.full_like(a_axis, np.nan)
    n_plus[valid] = (aR[valid] + np.sqrt(disc[valid])) / (2.0 * m)
    n_minus[valid] = (aR[valid] - np.sqrt(disc[valid])) / (2.0 * m)
    return n_plus, n_minus


# ----------------------------------------------------------------------------
# Tkinter application
# ----------------------------------------------------------------------------
class KlausmeierApp:
    SLIDER_SPECS = [
        ("Rainfall  a",           "a",  0.05, 5.0),
        ("Advection speed  v",    "v",  0.0,  300.0),
        ("Plant diffusion  Dn",   "Dn", 0.0,  5.0),
        ("Water diffusion  Dw",   "Dw", 0.0,  100.0),
        ("Plant mortality  m",    "m",  0.05, 1.5),
        ("Growth yield  R",       "R",  0.1,  3.0),
    ]

    def __init__(self, root):
        self.root = root
        root.title("Klausmeier (1999) semiarid vegetation model")
        root.geometry("1360x820")
        root.minsize(1180, 700)

        self.paused = False
        self.t_elapsed = 0.0
        self.slider_vars = {}
        self.value_labels = {}

        self._build_layout()
        self._build_plots()
        self.root.after(FRAME_INTERVAL_MS, self._tick)

    # -- layout -----------------------------------------------------------
    def _build_layout(self):
        style = ttk.Style()
        style.configure("Section.TLabel", font=("Helvetica", 11, "bold"))
        style.configure("Param.TLabel", font=("Helvetica", 10))
        style.configure("Note.TLabel", font=("Helvetica", 9))

        # Left panel: fixed width, generous enough for the longest label so
        # nothing gets clipped or wraps awkwardly.
        left = ttk.Frame(self.root, padding=(14, 14, 10, 14), width=340)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        right = ttk.Frame(self.root, padding=(4, 8, 8, 8))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_frame = right

        ttk.Label(left, text="Parameters", style="Section.TLabel").pack(anchor="w", pady=(0, 8))

        for label_text, key, vmin, vmax in self.SLIDER_SPECS:
            row = ttk.Frame(left)
            row.pack(fill=tk.X, pady=(0, 10))

            header = ttk.Frame(row)
            header.pack(fill=tk.X)
            ttk.Label(header, text=label_text, style="Param.TLabel").pack(side=tk.LEFT, anchor="w")
            val_lbl = ttk.Label(header, text=f"{params[key]:.3g}", style="Param.TLabel", width=7,
                                 anchor="e")
            val_lbl.pack(side=tk.RIGHT)
            self.value_labels[key] = val_lbl

            var = tk.DoubleVar(value=params[key])
            self.slider_vars[key] = var
            scale = ttk.Scale(row, from_=vmin, to=vmax, orient=tk.HORIZONTAL, variable=var,
                               command=lambda v, k=key: self._on_slider_change(k))
            scale.pack(fill=tk.X)

            rng_lbl = ttk.Label(row, text=f"[{vmin:g} - {vmax:g}]", style="Note.TLabel",
                                 foreground="#666666")
            rng_lbl.pack(anchor="w")

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=10)

        btn_row = ttk.Frame(left)
        btn_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(btn_row, text="Reset field", command=self._on_reset).pack(side=tk.LEFT,
                                                                              expand=True, fill=tk.X,
                                                                              padx=(0, 4))
        self.pause_btn = ttk.Button(btn_row, text="Pause", command=self._on_pause)
        self.pause_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        self.time_label = ttk.Label(left, text="t = 0.0", style="Param.TLabel")
        self.time_label.pack(anchor="w", pady=(4, 10))

        note = ("Reset field: reseeds the 2D domain near the current homogeneous\n"
                "equilibrium (plus small noise) so a fresh pattern can develop.\n\n"
                "Sliders act live on the running simulation and on the bifurcation\n"
                "curve below; the red dot on the bifurcation diagram tracks the\n"
                "spatial average of vegetation N over the whole 2D domain.")
        ttk.Label(left, text=note, style="Note.TLabel", wraplength=310, justify="left").pack(
            anchor="w")

    def _build_plots(self):
        self.fig = Figure(figsize=(9.2, 8.0), dpi=100)
        self.fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.08, hspace=0.32)
        self.ax_pattern = self.fig.add_subplot(2, 1, 1)
        self.ax_bifurc = self.fig.add_subplot(2, 1, 2)

        self.im = self.ax_pattern.imshow(N.T, origin="lower", cmap="YlGn",
                                          extent=[0, Lx, 0, Ly], vmin=0, vmax=3,
                                          interpolation="bilinear")
        self.cbar = self.fig.colorbar(self.im, ax=self.ax_pattern, fraction=0.046, pad=0.04)
        self.cbar.set_label("Plant biomass N")
        self.ax_pattern.set_xlabel("X (downhill direction)")
        self.ax_pattern.set_ylabel("Y")
        self.ax_pattern.set_title("Spatial vegetation pattern N(X, Y, t)")

        self.a_axis = np.linspace(0.05, 5.0, 400)
        n_plus0, n_minus0 = bifurcation_curves(self.a_axis, params["m"], params["R"])
        (self.line_stable,) = self.ax_bifurc.plot(self.a_axis, n_plus0, "g-", lw=2,
                                                   label="vegetated, stable")
        (self.line_unstable,) = self.ax_bifurc.plot(self.a_axis, n_minus0, "g--", lw=1.5,
                                                      label="vegetated, unstable")
        self.ax_bifurc.plot(self.a_axis, np.zeros_like(self.a_axis), color="saddlebrown", lw=2,
                             label="bare soil, stable")
        self.fold_line = self.ax_bifurc.axvline(2 * params["m"] / params["R"], color="gray",
                                                  ls=":", lw=1)
        (self.marker_point,) = self.ax_bifurc.plot([params["a"]], [np.nanmean(N)], "o",
                                                     color="crimson", ms=9, mec="k", zorder=5,
                                                     label="2D sim. (mean N)")
        self.ax_bifurc.set_xlim(0, 5)
        self.ax_bifurc.set_ylim(-0.2, 6)
        self.ax_bifurc.set_xlabel("Rainfall a")
        self.ax_bifurc.set_ylabel("Equilibrium N")
        self.ax_bifurc.set_title("Bifurcation diagram (non-spatial model)")
        self.ax_bifurc.legend(loc="upper left", fontsize=8, framealpha=0.9)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

    # -- callbacks ----------------------------------------------------------
    def _on_slider_change(self, key):
        val = self.slider_vars[key].get()
        params[key] = val
        self.value_labels[key].config(text=f"{val:.3g}")

        n_plus, n_minus = bifurcation_curves(self.a_axis, params["m"], params["R"])
        self.line_stable.set_ydata(n_plus)
        self.line_unstable.set_ydata(n_minus)
        self.fold_line.set_xdata([2 * params["m"] / params["R"]] * 2)
        self.canvas.draw_idle()

    def _on_reset(self):
        global W, N
        W, N = init_fields()
        self.t_elapsed = 0.0
        self.im.set_data(N.T)
        self.canvas.draw_idle()

    def _on_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="Resume" if self.paused else "Pause")

    # -- main loop ------------------------------------------------------------
    def _tick(self):
        if not self.paused:
            step()
            self.t_elapsed += DT * SUBSTEPS_PER_FRAME
            self.im.set_data(N.T)
            self.im.set_clim(vmin=0, vmax=max(1.0, N.max()))
            self.marker_point.set_data([params["a"]], [N.mean()])
            self.time_label.config(text=f"t = {self.t_elapsed:.1f}")
            self.canvas.draw_idle()
        self.root.after(FRAME_INTERVAL_MS, self._tick)


if __name__ == "__main__":
    root = tk.Tk()
    app = KlausmeierApp(root)
    root.mainloop()