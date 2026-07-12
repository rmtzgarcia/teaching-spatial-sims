"""
Plant Community Simulation (top-hat competition kernel)
===========================================================
Exact implementation of the individual-based model in Surendran, Pinto-Ramos,
Menezes & Martinez-Garcia, "Spatial moment dynamics and biomass density
equations provide complementary, yet limited, descriptions of pattern
formation in individual-based simulations", Physica D 477 (2025) 134703,
Section 2.1, with the top-hat competition kernel (Eq. 2.3, second case).

Individuals live on a square domain of side `DOMAIN_L` with periodic boundary
conditions. The population is simulated as an exact continuous-time
birth-death process (Gillespie's stochastic simulation algorithm):
  - each individual reproduces at constant rate `b` (Eq. 2.1), dispersing its
    offspring at an offset drawn from an isotropic Normal(0, s^2) kernel;
  - each individual dies at rate d + g * sum_j phi_C(x_i - x_j) (Eq. 2.2),
    where phi_C is a top-hat competition kernel of range `r_c`: 1/(pi r_c^2)
    if the distance is within `r_c`, else 0 (normalized to integrate to 1).

Interface: Reset / Start buttons, b / d / g entry boxes (intrinsic birth and
death rates and competition strength, held fixed in the paper's Table 1),
r_c / s sliders (competition and dispersal range, the two parameters the
paper systematically varies), a population-over-time plot with the
mean-field (non-spatial) prediction N_h = (b-d)/g * L^2, a spatial view of
the simulated patch, and a pair-correlation-function C(r) diagnostic (this
is the kernel choice that produces the periodic/regular aggregated patterns
described in Section 3.2 of the paper).

Requires: numpy, matplotlib (both commonly pre-installed; `pip install numpy
matplotlib --break-system-packages` if missing).
"""

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

DOMAIN_L = 20.0  # square domain side length; periodic boundary conditions

# Table 1 defaults
DEFAULT_B = 3.05    # intrinsic birth rate
DEFAULT_D = 0.305   # intrinsic death rate
DEFAULT_G = 1.0     # competition strength
DEFAULT_RC = 2.2    # competition range (slider default; paper sweeps 0-4)
DEFAULT_S = 0.15    # dispersal range (slider default; paper sweeps 0-1.25)

MAX_N_INIT = 4000  # safety cap so a runaway parameter choice can't hang the GUI
EVENTS_PER_ANIMATION_FRAME = 150


def competition_kernel(dist, r_c):
    """Top-hat competition kernel (Eq. 2.3, second case), normalized to
    integrate to 1 over the plane: phi_C(r) = 1/(pi r_c^2) if r <= r_c, else 0.
    """
    return np.where(dist <= r_c, 1.0 / (np.pi * r_c ** 2), 0.0)


def predicted_equilibrium(b, d, g):
    """Mean-field (well-mixed) equilibrium population size,
    N_h = (b - d) / g * L^2 (Surendran et al. 2025, Section 2.3 and Fig. 1B).

    Unlike a finite, non-periodic domain, this does not depend on the
    competition/dispersal ranges r_c, s: both kernels are normalized to
    integrate to 1, and the periodic domain has no edges to correct for.
    """
    if g <= 0:
        return float("nan")
    density_h = (b - d) / g
    if density_h <= 0:
        return 0.0
    return density_h * DOMAIN_L ** 2


class PlantCommunityModel:
    """Exact continuous-time spatial birth-death IBM (Surendran et al. 2025,
    Section 2.1), simulated via the Gillespie stochastic simulation
    algorithm with an incrementally-updated competition sum so each event
    costs O(N) instead of recomputing the full O(N^2) pairwise kernel.
    """

    def __init__(self, r_c, s, b, d, g, seed=None, n_init=None):
        self.rng = np.random.default_rng(seed)
        self.r_c = r_c
        self.s = s
        self.b = b
        self.d = d
        self.g = g

        if n_init is None:
            n_init = int(np.clip(round(predicted_equilibrium(b, d, g)), 10, MAX_N_INIT))
        self.positions = self.rng.uniform(0.0, DOMAIN_L, size=(n_init, 2))

        self.t = 0.0
        self.t_history = [0.0]
        self.history = [n_init]

        self._comp_sum = self._initial_competition_sums()

    # -- periodic geometry (minimum-image convention) ------------------------
    def _periodic_diff(self, a, b):
        diff = a - b
        return diff - DOMAIN_L * np.round(diff / DOMAIN_L)

    def _dist_to_point(self, points, point):
        diff = self._periodic_diff(points, point)
        return np.sqrt((diff ** 2).sum(axis=1))

    def _full_pairwise_dist(self, points):
        diff = points[:, None, :] - points[None, :, :]
        diff -= DOMAIN_L * np.round(diff / DOMAIN_L)
        return np.sqrt((diff ** 2).sum(axis=2))

    def _initial_competition_sums(self):
        n = len(self.positions)
        if n == 0:
            return np.zeros(0)
        dist = self._full_pairwise_dist(self.positions)
        k = competition_kernel(dist, self.r_c)
        np.fill_diagonal(k, 0.0)
        return k.sum(axis=1)

    # -- Gillespie SSA --------------------------------------------------------
    def step_events(self, n_events=1):
        """Advance the exact continuous-time birth-death process by up to
        `n_events` events (fewer if the population goes extinct)."""
        for _ in range(n_events):
            n = len(self.positions)
            if n == 0:
                break

            death_rates = self.d + self.g * self._comp_sum
            total_birth = self.b * n
            total_death = death_rates.sum()
            total = total_birth + total_death
            if total <= 0:
                break

            self.t += self.rng.exponential(1.0 / total)

            if self.rng.random() * total < total_birth:
                self._do_birth()
            else:
                self._do_death(death_rates, total_death)

            self.t_history.append(self.t)
            self.history.append(len(self.positions))

    def _do_birth(self):
        n = len(self.positions)
        parent = self.positions[self.rng.integers(n)]
        offset = self.rng.normal(0.0, self.s, size=2)
        new_pos = (parent + offset) % DOMAIN_L

        k_vals = competition_kernel(
            self._dist_to_point(self.positions, new_pos), self.r_c
        )
        self._comp_sum += k_vals  # existing individuals gain a neighbor
        self.positions = np.vstack([self.positions, new_pos])
        self._comp_sum = np.append(self._comp_sum, k_vals.sum())

    def _do_death(self, death_rates, total_death):
        cum = np.cumsum(death_rates)
        die_idx = int(np.searchsorted(cum, self.rng.random() * total_death))
        die_idx = min(die_idx, len(self.positions) - 1)
        dead_pos = self.positions[die_idx]

        k_vals = competition_kernel(
            self._dist_to_point(self.positions, dead_pos), self.r_c
        )
        self._comp_sum -= k_vals
        self.positions = np.delete(self.positions, die_idx, axis=0)
        self._comp_sum = np.delete(self._comp_sum, die_idx)


def radial_correlation_function(positions, domain_l, r_max=None, n_bins=40):
    """Estimate C(r) (Eq. 2.7 in Surendran et al. 2025) for a point pattern on
    a periodic (torus) square domain of side `domain_l`.

    C(r) = 1 for a fully random (Poisson / well-mixed) pattern; C(r) > 1 at
    small r indicates clustering, C(r) < 1 indicates over-dispersion
    (regularity/self-thinning). Because the domain is periodic there is no
    boundary to correct for: every point serves as a focal point, and
    distances use the minimum-image convention.

    Returns (r_centers, C); raises ValueError if there are too few points.
    """
    positions = np.asarray(positions)
    n = len(positions)
    if n < 2:
        raise ValueError("Need at least 2 plants to estimate C(r).")
    if r_max is None:
        r_max = domain_l / 2  # largest radius with an unambiguous periodic distance

    area = domain_l ** 2
    density = n / area  # Z1, the mean plant density

    diff = positions[:, None, :] - positions[None, :, :]
    diff -= domain_l * np.round(diff / domain_l)
    dist = np.sqrt((diff ** 2).sum(axis=2))

    edges = np.linspace(0, r_max, n_bins + 1)
    dr = edges[1] - edges[0]
    r_centers = 0.5 * (edges[:-1] + edges[1:])

    counts, _ = np.histogram(dist[dist > 0], bins=edges)

    # expected neighbor count per annulus under complete spatial randomness
    expected = n * density * 2 * np.pi * r_centers * dr

    return r_centers, counts / expected


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------


class SliderRow(ttk.Frame):
    """A labeled horizontal slider with a live numeric readout, NetLogo-style."""

    def __init__(self, parent, label, frm, to, default, resolution=0.01):
        super().__init__(parent)
        self.var = tk.DoubleVar(value=default)
        self.scale = ttk.Scale(
            self, from_=frm, to=to, orient="horizontal", variable=self.var,
            command=self._on_move,
        )
        self.scale.pack(fill="x", expand=True, side="top")

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", side="top")
        ttk.Label(bottom, text=label).pack(side="left")
        self.value_label = ttk.Label(bottom, text=f"{default:.2f}")
        self.value_label.pack(side="right")
        self.resolution = resolution

    def _on_move(self, _evt=None):
        val = round(self.var.get() / self.resolution) * self.resolution
        self.value_label.config(text=f"{val:.2f}")

    def get(self):
        return self.var.get()


class PlantCommunityApp:
    def __init__(self, root):
        self.root = root
        root.title("Plant Community Simulation")
        root.configure(bg="#ececec")

        self.model = None
        self.running = False

        self._build_layout()
        self.reset()

    # -- layout -----------------------------------------------------------
    def _build_layout(self):
        left = ttk.Frame(self.root)
        left.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # --- Toolbar: Reset / Start / b / d / g ---
        toolbar = ttk.Frame(left)
        toolbar.pack(fill="x", pady=(0, 8))

        self.reset_btn = ttk.Button(toolbar, text="Reset", command=self.reset)
        self.reset_btn.grid(row=0, column=0, padx=(0, 6), sticky="nsew")

        self.start_btn = ttk.Button(toolbar, text="Start!", command=self.toggle_run)
        self.start_btn.grid(row=0, column=1, padx=(0, 12), sticky="nsew")

        ttk.Label(toolbar, text="b").grid(row=0, column=2, sticky="w")
        self.b_var = tk.StringVar(value=str(DEFAULT_B))
        ttk.Entry(toolbar, textvariable=self.b_var, width=6).grid(
            row=0, column=3, padx=(4, 10)
        )

        ttk.Label(toolbar, text="d").grid(row=0, column=4, sticky="w")
        self.d_var = tk.StringVar(value=str(DEFAULT_D))
        ttk.Entry(toolbar, textvariable=self.d_var, width=6).grid(
            row=0, column=5, padx=(4, 10)
        )

        ttk.Label(toolbar, text="g").grid(row=0, column=6, sticky="w")
        self.g_var = tk.StringVar(value=str(DEFAULT_G))
        ttk.Entry(toolbar, textvariable=self.g_var, width=6).grid(
            row=0, column=7, padx=(4, 0)
        )

        self.corr_btn = ttk.Button(
            toolbar, text="Pair correlation C(r)", command=self.show_pair_correlation
        )
        self.corr_btn.grid(row=0, column=8, padx=(12, 0), sticky="nsew")

        # --- Sliders: competition range r_c, dispersal range s ---
        self.rc_slider = SliderRow(left, "r_c", 0.01, 4, DEFAULT_RC)
        self.rc_slider.pack(fill="x", pady=(0, 4))

        self.s_slider = SliderRow(left, "s", 0.01, 1.25, DEFAULT_S)
        self.s_slider.pack(fill="x", pady=(0, 8))

        # --- Population plot ---
        self.fig = Figure(figsize=(4.6, 4.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Total population size")
        self.ax.set_xlabel("t")
        self.ax.set_ylabel("# individuals")
        (self.line,) = self.ax.plot([], [], color="black", linewidth=1, label="population")
        self.pred_line = self.ax.axhline(
            0, color="red", linestyle="--", linewidth=1, label="non-spatial limit N_h"
        )
        self.ax.legend(loc="upper right", fontsize=8)

        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- Spatial view (right side): brown ground, green plants ---
        right = ttk.Frame(self.root)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)

        self.count_var = tk.StringVar(value="Plants: 0")
        ttk.Label(
            right, textvariable=self.count_var, font=("TkDefaultFont", 11, "bold"),
        ).pack(fill="x", pady=(0, 4))

        self.canvas_size = 560
        self.canvas = tk.Canvas(
            right, width=self.canvas_size, height=self.canvas_size,
            bg="#5a3d2b", highlightthickness=1, highlightbackground="black",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def _on_canvas_resize(self, event):
        self.canvas_w = event.width
        self.canvas_h = event.height
        self._redraw_spatial()

    # -- world <-> pixel mapping -------------------------------------------
    def _to_canvas(self, xy):
        w = getattr(self, "canvas_w", self.canvas_size)
        h = getattr(self, "canvas_h", self.canvas_size)
        px = xy[:, 0] / DOMAIN_L * w
        py = h - xy[:, 1] / DOMAIN_L * h
        return px, py

    # -- controls -----------------------------------------------------------
    def reset(self):
        self.running = False
        self.start_btn.config(text="Start!")
        try:
            b = float(self.b_var.get())
        except ValueError:
            b = DEFAULT_B
        try:
            d = float(self.d_var.get())
        except ValueError:
            d = DEFAULT_D
        try:
            g = float(self.g_var.get())
        except ValueError:
            g = DEFAULT_G

        self.model = PlantCommunityModel(
            r_c=self.rc_slider.get(),
            s=self.s_slider.get(),
            b=b,
            d=d,
            g=g,
        )
        self._redraw()

    def toggle_run(self):
        self.running = not self.running
        self.start_btn.config(text="Stop" if self.running else "Start!")
        if self.running:
            self._loop()

    def _loop(self):
        if not self.running:
            return
        # pick up any live slider/entry changes each frame, NetLogo-style
        self.model.r_c = self.rc_slider.get()
        self.model.s = self.s_slider.get()
        try:
            self.model.b = float(self.b_var.get())
        except ValueError:
            pass
        try:
            self.model.d = float(self.d_var.get())
        except ValueError:
            pass
        try:
            self.model.g = float(self.g_var.get())
        except ValueError:
            pass

        self.model.step_events(EVENTS_PER_ANIMATION_FRAME)
        self._redraw()
        self.root.after(40, self._loop)

    def show_pair_correlation(self):
        try:
            r, c = radial_correlation_function(self.model.positions, DOMAIN_L)
        except ValueError as exc:
            messagebox.showinfo("Pair correlation C(r)", str(exc))
            return

        win = tk.Toplevel(self.root)
        win.title("Pair correlation function C(r)")

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.axhline(1, color="gray", linestyle="--", linewidth=1, label="C(r) = 1 (random)")
        ax.plot(r, c, color="black", linewidth=1.5, label="C(r)")
        ax.set_xlabel("r")
        ax.set_ylabel("C(r)")
        ax.set_title(f"Pair correlation (t={self.model.t:.2f}, N={len(self.model.positions)})")
        ax.legend(loc="best", fontsize=8)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    # -- drawing --------------------------------------------------------------
    def _redraw_spatial(self):
        self.canvas.delete("plant")
        if self.model is None:
            return
        pos = self.model.positions
        if len(pos):
            px, py = self._to_canvas(pos)
            r = 3
            for x, y in zip(px, py):
                self.canvas.create_oval(
                    x - r, y - r, x + r, y + r,
                    fill="#3aa63a", outline="", tags="plant",
                )
        self.count_var.set(f"Plants: {len(pos)}")

    def _redraw(self):
        self._redraw_spatial()

        # population plot
        self.line.set_data(self.model.t_history, self.model.history)
        n_star = predicted_equilibrium(self.model.b, self.model.d, self.model.g)
        self.pred_line.set_ydata([n_star, n_star])
        self.ax.relim()
        self.ax.autoscale_view()
        self.plot_canvas.draw_idle()


def main():
    root = tk.Tk()
    PlantCommunityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
