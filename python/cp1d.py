"""
1D Contact Process — translated from cp1d.f90 / dranxor.f90
=============================================================

Faithful Python translation of the Fortran contact-process simulation.
Physics / algorithm (unchanged from the Fortran):

  - N sites on a ring (periodic boundary conditions, via left/right lookup
    tables exactly as in the Fortran code).
  - Each site is empty (0) or occupied (1). The list `soc` holds the
    indices of all occupied sites so that "pick a uniformly random
    occupied site" is O(1) (same trick as the Fortran `soc` array).
  - One "event": pick a random occupied site.
      * with probability (1-p): it dies -> removed from soc via the
        swap-with-last-element trick, exactly as
            s(soc(ic)) = 0
            soc(ic) = soc(oc); oc = oc - 1
        in the Fortran source.
      * with probability p: it attempts a birth. It picks its left or
        right neighbour with 50/50 chance, and if that neighbour is
        empty, the neighbour becomes occupied and is appended to soc.
  - `t` counts events (exactly like the Fortran `t = t + 1.d0`), so it is
    NOT physical/continuous time — it's an event counter. One "sweep"
    below means N such events, i.e. one event per site on average.
  - The all-empty configuration is absorbing (oc = 0): once reached the
    process can never recover, matching the Fortran `if(oc.eq.0) go to 1`.
  - The parameter `p` is the birth-probability control from the Fortran code
    (called "birth probability" in the task): p close to 1 favours birth events,
    p close to 0 favours death. The classic 1D contact process has a
    critical point p_c ≈ 0.6494: below it the population always dies out
    (for large N); above it it can survive indefinitely.

Randomness: the Fortran `dranxor` module is a bespoke lagged-Fibonacci
generator used purely as a fast RNG. It carries no physics, so it is
replaced here with NumPy's default_rng (PCG64) — a modern, well-tested
generator that plays the same role as `dran_u` / `i_dran`.

GUI: built with native Tkinter widgets (matching spatial-logistic.py) rather
than matplotlib's own Slider/Button + FuncAnimation, which are unreliable
about receiving clicks once an animation timer is running. Two live panels:
  1. top panel: fraction of occupied sites (density) vs. sweep.
  2. bottom panel: a space-time raster — a matrix where each row is a
     snapshot of the whole 1D lattice at a given sweep, so the image
     shows how the lattice pattern evolves as new rows are added.
A slider lets you change the birth probability p while the simulation runs.
Pause/Resume and Reset buttons are provided as well. `--sweeps-per-frame`
lets the simulation advance several sweeps between redraws, so it isn't
bottlenecked on plot/canvas rendering speed.

Usage:
    python cp1d.py
    python cp1d.py --N 500 --rho0 0.35 --p 0.85 --rows 300 --sweeps-per-frame 10
"""

import argparse

import numpy as np
import tkinter as tk
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ContactProcess1D:
    """Event-driven 1D contact process on a ring, list-based (O(1) per event)."""

    def __init__(self, N=300, rho0=0.35, p=0.85, seed=None):
        self.N = N
        self.rho0 = rho0
        self.p = p
        self.rng = np.random.default_rng(seed)

        # periodic boundary lookup tables (equivalent to left(:)/right(:) in Fortran)
        self.left = np.arange(N) - 1
        self.left[0] = N - 1
        self.right = np.arange(N) + 1
        self.right[N - 1] = 0

        self.reset()

    def reset(self, rho0=None):
        """Reinitialize the lattice at random with density rho0, clear time."""
        if rho0 is not None:
            self.rho0 = rho0
        N = self.N
        self.s = (self.rng.random(N) < self.rho0).astype(np.int8)
        self.soc = list(np.flatnonzero(self.s))
        self.oc = len(self.soc)
        self.t = 0.0

    def step(self):
        """Perform a single contact-process event. Returns False if extinct."""
        if self.oc == 0:
            return False

        ic = self.rng.integers(0, self.oc)
        site = self.soc[ic]

        if self.rng.random() >= self.p:
            # death event
            self.s[site] = 0
            last = self.oc - 1
            self.soc[ic] = self.soc[last]
            self.soc.pop()
            self.oc -= 1
        else:
            # birth attempt on a random neighbour
            if self.rng.random() < 0.5:
                nb = self.left[site]
            else:
                nb = self.right[site]
            if self.s[nb] == 0:
                self.s[nb] = 1
                self.soc.append(nb)
                self.oc += 1

        self.t += 1.0
        return self.oc > 0

    def sweep(self, n_events=None):
        """Run n_events events (default: N, i.e. one sweep). Returns False if extinct."""
        if n_events is None:
            n_events = self.N
        alive = True
        for _ in range(n_events):
            alive = self.step()
            if not alive:
                break
        return alive


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------


class SliderRow(ttk.Frame):
    """A labeled horizontal slider with a live numeric readout, NetLogo-style."""

    def __init__(self, parent, label, frm, to, default, resolution=0.001, command=None):
        super().__init__(parent)
        self.command = command
        self.var = tk.DoubleVar(value=default)
        self.scale = ttk.Scale(
            self, from_=frm, to=to, orient="horizontal", variable=self.var,
            command=self._on_move,
        )
        self.scale.pack(fill="x", expand=True, side="top")

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", side="top")
        ttk.Label(bottom, text=label).pack(side="left")
        self.value_label = ttk.Label(bottom, text=f"{default:.3f}")
        self.value_label.pack(side="right")
        self.resolution = resolution

    def _on_move(self, _evt=None):
        val = round(self.var.get() / self.resolution) * self.resolution
        self.value_label.config(text=f"{val:.3f}")
        if self.command is not None:
            self.command(val)

    def get(self):
        return self.var.get()


class ContactProcessApp:
    def __init__(self, root, cp, max_rows=300, window_sweeps=400,
                 sweeps_per_frame=1, interval_ms=40):
        self.root = root
        self.cp = cp
        self.max_rows = max_rows
        self.window_sweeps = window_sweeps
        self.sweeps_per_frame = sweeps_per_frame
        self.interval_ms = interval_ms

        root.title("1D Lattice Logistic Model")
        root.configure(bg="#ececec")

        self.running = True
        self.extinct = False
        self.sweep_count = 0

        self.sweep_hist = [0]
        self.rho_hist = [cp.oc / cp.N]
        self.history = np.zeros((max_rows, cp.N), dtype=np.int8)
        self.history[0] = cp.s

        self._build_layout()
        self._loop()

    # -- layout -----------------------------------------------------------
    def _build_layout(self):
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True, padx=8, pady=8)

        # --- Toolbar: Pause / Reset / status ---
        toolbar = ttk.Frame(container)
        toolbar.pack(fill="x", pady=(0, 8))

        self.pause_btn = ttk.Button(toolbar, text="Pause", command=self.toggle_run)
        self.pause_btn.grid(row=0, column=0, padx=(0, 6), sticky="nsew")

        self.reset_btn = ttk.Button(toolbar, text="Reset", command=self.reset)
        self.reset_btn.grid(row=0, column=1, padx=(0, 12), sticky="nsew")

        self.status_var = tk.StringVar()
        ttk.Label(
            toolbar, textvariable=self.status_var, font=("TkDefaultFont", 10),
        ).grid(row=0, column=2, sticky="w")

        # --- Slider: birth probability p ---
        self.p_slider = SliderRow(
            container, "birth probability p", 0.0, 1.0, self.cp.p,
            resolution=0.001, command=self._on_p_change,
        )
        self.p_slider.pack(fill="x", pady=(0, 8))

        # --- Plots ---
        self.fig = Figure(figsize=(7, 7), dpi=100)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 1.6], hspace=0.35)
        self.ax_rho = self.fig.add_subplot(gs[0])
        self.ax_lat = self.fig.add_subplot(gs[1])

        (self.line,) = self.ax_rho.plot([], [], lw=1.5, color="tab:blue")
        self.ax_rho.set_xlabel("sweep (≈ events / N)")
        self.ax_rho.set_ylabel("occupied fraction ρ(t)")
        self.ax_rho.set_ylim(-0.02, 1.02)

        self.img = self.ax_lat.imshow(
            self.history, aspect="auto", cmap="Greys", interpolation="nearest",
            vmin=0, vmax=1,
        )
        self.ax_lat.set_xlabel("site index")
        self.ax_lat.set_ylabel("sweep (newest at top)")
        self.ax_lat.set_yticks([])

        self.canvas = FigureCanvasTkAgg(self.fig, master=container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # -- controls -----------------------------------------------------------
    def _on_p_change(self, val):
        self.cp.p = val

    def toggle_run(self):
        self.running = not self.running
        self.pause_btn.config(text="Resume" if not self.running else "Pause")

    def reset(self):
        self.cp.reset()
        self.history[:] = 0
        self.history[0] = self.cp.s
        self.sweep_hist = [0]
        self.rho_hist = [self.cp.oc / self.cp.N]
        self.sweep_count = 0
        self.extinct = False
        self.running = True
        self.pause_btn.config(text="Pause")
        self._redraw()

    def _loop(self):
        if self.running and not self.extinct:
            for _ in range(self.sweeps_per_frame):
                alive = self.cp.sweep(self.cp.N)  # one sweep = N events
                self.sweep_count += 1

                self.history[1:] = self.history[:-1]
                self.history[0] = self.cp.s

                self.sweep_hist.append(self.sweep_count)
                self.rho_hist.append(self.cp.oc / self.cp.N)

                if not alive:
                    self.extinct = True
                    break

            if len(self.sweep_hist) > self.window_sweeps:
                self.sweep_hist = self.sweep_hist[-self.window_sweeps:]
                self.rho_hist = self.rho_hist[-self.window_sweeps:]

        self._redraw()
        self.root.after(self.interval_ms, self._loop)

    # -- drawing --------------------------------------------------------------
    def _redraw(self):
        self.line.set_data(self.sweep_hist, self.rho_hist)
        self.ax_rho.set_xlim(max(0, self.sweep_hist[0]), max(self.sweep_hist[-1], 10))

        self.img.set_data(self.history)

        tag = " -- EXTINCT (absorbing state, press Reset)" if self.extinct else ""
        self.status_var.set(
            f"p={self.cp.p:.3f}  sweep={self.sweep_count}  "
            f"rho={self.cp.oc / self.cp.N:.3f}{tag}"
        )
        self.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="1D contact process (translated from cp1d.f90)")
    parser.add_argument("--N", type=int, default=300, help="number of lattice sites")
    parser.add_argument("--rho0", type=float, default=0.35, help="initial density")
    parser.add_argument("--p", type=float, default=0.85, help="initial birth probability p")
    parser.add_argument("--rows", type=int, default=300, help="rows kept in the space-time raster")
    parser.add_argument("--sweeps-per-frame", type=int, default=1,
                         help="simulated sweeps advanced between redraws")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    args = parser.parse_args()

    cp = ContactProcess1D(N=args.N, rho0=args.rho0, p=args.p, seed=args.seed)
    root = tk.Tk()
    ContactProcessApp(root, cp, max_rows=args.rows, sweeps_per_frame=args.sweeps_per_frame)
    root.mainloop()


if __name__ == "__main__":
    main()
