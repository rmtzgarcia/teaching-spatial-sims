"""
Interactive lattice simulation of the spatial "allelopathy" (colicin) model
proposed in:

    Durrett, R. & Levin, S. A. (1997). Allelopathy in Spatially Distributed
    Populations. J. theor. Biol. 185, 165-171.

MODEL (Section 3 of the paper)
-------------------------------
Space is a periodic L x L grid (torus). Each site x has a state
    xi_t(x) in {0, 1, 2}
        0 = vacant
        1 = occupied by a colicin PRODUCER          -> drawn RED
        2 = occupied by a colicin-SENSITIVE cell,
            i.e. a "non-producer"                    -> drawn BLUE

Let f1(x), f2(x) be the fraction of x's neighbors (using either the 4
nearest neighbors N0, or the 8 nearest+diagonal neighbors N1) that are in
state 1 or 2. The process evolves in continuous time with transition rates:

    birth   0 -> 1   at rate  b1 * f1(x)
    birth   0 -> 2   at rate  b2 * f2(x)
    death   1 -> 0   at rate  d1
    death   2 -> 0   at rate  d2 + gamma * f1(x)

In words: each empty site is colonized by a neighboring type at a rate
proportional to how many neighbors of that type it has; producers die at a
constant background rate d1; sensitive cells die at their background rate
d2 *plus* an extra colicin-induced hazard gamma proportional to the
fraction of neighboring producers.

The paper itself defines "rate lambda" operationally: the probability of
the event happening in a short time interval of length dt is lambda*dt.
That is exactly how this script simulates the process -- at every time
step of size dt we draw one uniform random number per site and use it to
decide (vectorized with numpy) which single event, if any, fires at that
site. This reproduces the qualitative phenomena reported in the paper:
with the default parameters (d1=d2=1, b1=3, b2=4, gamma=3) colicin
producers, although rare initially, form growing clumps and take over the
grid (Figs 2-4); reducing gamma to 1 reverses the outcome and the
sensitive strain wins (Fig 5).

GUI
---
A matplotlib-widgets interface (sliders + buttons + radio buttons) lets you
change all model parameters live, pick the neighborhood (N0 or N1), and
play/pause/reset the simulation while watching the lattice (red vs. blue)
and the density-vs-time curves (reproducing Fig. 2 of the paper) update in
real time.

Requirements: numpy, matplotlib
    pip install numpy matplotlib

Run:
    python colicin_lattice_model.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation

# --------------------------------------------------------------------------
# Neighborhoods (relative (row, col) shifts), as defined in the paper.
# --------------------------------------------------------------------------
N0_SHIFTS = [(1, 0), (-1, 0), (0, 1), (0, -1)]                       # 4 nearest neighbors
N1_SHIFTS = N0_SHIFTS + [(1, 1), (1, -1), (-1, 1), (-1, -1)]         # + 4 diagonal neighbors


# --------------------------------------------------------------------------
# The stochastic spatial process itself (model logic, no plotting here).
# --------------------------------------------------------------------------
class ColicinLatticeModel:
    def __init__(self, size=100, b1=3.0, b2=4.0, d1=1.0, d2=1.0, gamma=3.0,
                 p1_init=0.01, p2_init=0.50, neighborhood="N0", dt=0.05, seed=None):
        self.size = int(size)
        self.b1, self.b2 = float(b1), float(b2)
        self.d1, self.d2 = float(d1), float(d2)
        self.gamma = float(gamma)
        self.p1_init, self.p2_init = float(p1_init), float(p2_init)
        self.neighborhood = neighborhood
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)

        self.time = 0.0
        self.history_t = []
        self.history_f1 = []
        self.history_f2 = []
        self.grid = None
        self.reset()

    def reset(self):
        """Re-seed the grid from an i.i.d. 'product measure' initial condition,
        exactly as described in Section 4 of the paper."""
        L = self.size
        r = self.rng.random((L, L))
        grid = np.zeros((L, L), dtype=np.int8)
        grid[r < self.p1_init] = 1
        grid[(r >= self.p1_init) & (r < self.p1_init + self.p2_init)] = 2
        self.grid = grid
        self.time = 0.0
        self.history_t = []
        self.history_f1 = []
        self.history_f2 = []
        self._record()

    def _shifts(self):
        return N1_SHIFTS if self.neighborhood == "N1" else N0_SHIFTS

    def _neighbor_fraction(self, mask):
        """Fraction of neighbors (under the chosen neighborhood) for which
        `mask` is True, computed for every site at once via periodic
        (torus) shifts -- this is what gives periodic boundary conditions."""
        shifts = self._shifts()
        total = np.zeros(self.grid.shape, dtype=float)
        for dx, dy in shifts:
            total += np.roll(np.roll(mask, dx, axis=0), dy, axis=1)
        return total / len(shifts)

    def step(self):
        """Advance the process by one time increment dt."""
        grid = self.grid
        dt = self.dt

        is1 = grid == 1
        is2 = grid == 2
        f1 = self._neighbor_fraction(is1)
        f2 = self._neighbor_fraction(is2)

        # Event probabilities in this dt window (rate * dt, as defined in the paper).
        p_b1 = np.clip(self.b1 * f1 * dt, 0.0, 1.0)
        p_b2 = np.clip(self.b2 * f2 * dt, 0.0, 1.0)
        p_d1 = min(self.d1 * dt, 1.0)
        p_d2 = np.clip((self.d2 + self.gamma * f1) * dt, 0.0, 1.0)

        r = self.rng.random(grid.shape)
        new_grid = grid.copy()

        vacant = grid == 0
        new_grid[vacant & (r < p_b1)] = 1
        new_grid[vacant & (r >= p_b1) & (r < p_b1 + p_b2)] = 2

        producer = grid == 1
        new_grid[producer & (r < p_d1)] = 0

        sensitive = grid == 2
        new_grid[sensitive & (r < p_d2)] = 0

        self.grid = new_grid
        self.time += dt
        self._record()

    def _record(self):
        n = self.size * self.size
        self.history_t.append(self.time)
        self.history_f1.append(np.count_nonzero(self.grid == 1) / n)
        self.history_f2.append(np.count_nonzero(self.grid == 2) / n)


# --------------------------------------------------------------------------
# The interactive matplotlib application.
# --------------------------------------------------------------------------
class ColicinApp:
    def __init__(self):
        self.model = ColicinLatticeModel()
        self.running = False
        self.steps_per_frame = 1

        self.fig = plt.figure(figsize=(13, 8))
        self.fig.suptitle(
            "Durrett & Levin (1997) spatial colicin model    "
            "RED = colicin producer   BLUE = non-producer (sensitive)   white = vacant",
            fontsize=10.5,
        )

        # --- left column: all parameter-scan controls -------------------------
        # col_x leaves room for slider labels (drawn to the left of the bar)
        # so long ones like "gamma (colicin toxicity)" don't clip at the figure edge.
        col_x, col_w = 0.18, 0.21

        # --- sliders (single stacked column) ------------------------------
        row_y = [0.90 - i * 0.038 for i in range(10)]

        def add_slider(y, label, vmin, vmax, vinit, step=None):
            ax = self.fig.add_axes([col_x, y, col_w, 0.02])
            return Slider(ax, label, vmin, vmax, valinit=vinit, valstep=step)

        self.s_b1 = add_slider(row_y[0], "b1 (producer birth)", 0.0, 8.0, self.model.b1)
        self.s_b2 = add_slider(row_y[1], "b2 (sensitive birth)", 0.0, 8.0, self.model.b2)
        self.s_d1 = add_slider(row_y[2], "d1 (producer death)", 0.0, 4.0, self.model.d1)
        self.s_d2 = add_slider(row_y[3], "d2 (sensitive death)", 0.0, 4.0, self.model.d2)
        self.s_gamma = add_slider(row_y[4], "gamma (colicin toxicity)", 0.0, 8.0, self.model.gamma)
        self.s_p1 = add_slider(row_y[5], "init density (1's)", 0.0, 1.0, self.model.p1_init)
        self.s_p2 = add_slider(row_y[6], "init density (2's)", 0.0, 1.0, self.model.p2_init)
        self.s_dt = add_slider(row_y[7], "dt (time step)", 0.005, 0.2, self.model.dt)
        self.s_speed = add_slider(row_y[8], "steps / frame", 1, 20, self.steps_per_frame, step=1)
        self.s_size = add_slider(row_y[9], "grid size (on reset)", 20, 200, self.model.size, step=10)

        for s in (self.s_b1, self.s_b2, self.s_d1, self.s_d2, self.s_gamma, self.s_dt, self.s_speed):
            s.on_changed(self._on_live_param_change)

        # --- neighborhood choice + buttons ---------------------------------
        # Centered as a block: horizontally on the sliders' true visual span
        # (their labels are drawn to the *left* of col_x, so the block's
        # center isn't col_x + col_w/2 but shifted left of it), and
        # vertically on the density-vs-time plot next to it.
        block_x, block_w = 0.113, col_w

        self.ax_radio = self.fig.add_axes([block_x, 0.283, block_w, 0.14])
        self.ax_radio.set_title("Neighborhood", fontsize=9)
        self.radio = RadioButtons(self.ax_radio, ("4 neighbors (N0)", "8 neighbors (N1)"), active=0)
        self.radio.on_clicked(self._on_neighborhood_change)

        # --- buttons (stacked, narrow column) -------------------------------
        self.ax_play = self.fig.add_axes([block_x, 0.208, block_w, 0.045])
        self.btn_play = Button(self.ax_play, "Play")
        self.btn_play.on_clicked(self._on_play_pause)

        self.ax_reset = self.fig.add_axes([block_x, 0.153, block_w, 0.045])
        self.btn_reset = Button(self.ax_reset, "Reset")
        self.btn_reset.on_clicked(self._on_reset)

        self.ax_step = self.fig.add_axes([block_x, 0.098, block_w, 0.045])
        self.btn_step = Button(self.ax_step, "Single step")
        self.btn_step.on_clicked(self._on_single_step)

        self.info_text = self.fig.text(block_x, 0.045, "", fontsize=9)

        # --- right column: the two graphical widgets, stacked -------------------
        self.ax_lattice = self.fig.add_axes([0.46, 0.52, 0.48, 0.40])
        cmap = ListedColormap(["white", "red", "blue"])
        self.im = self.ax_lattice.imshow(
            self.model.grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest"
        )
        self.ax_lattice.set_xticks([])
        self.ax_lattice.set_yticks([])
        self.ax_lattice.set_title("t = 0.00")

        # --- density-vs-time plot (reproduces Fig. 2) -------------------------
        self.ax_density = self.fig.add_axes([0.46, 0.06, 0.48, 0.40])
        (self.line1,) = self.ax_density.plot([], [], color="red", lw=1.5, label="producers (1's)")
        (self.line2,) = self.ax_density.plot([], [], color="blue", lw=1.5, label="non-producers (2's)")
        self.ax_density.set_xlim(0, 50)
        self.ax_density.set_ylim(0, 1)
        self.ax_density.set_xlabel("time")
        self.ax_density.set_ylabel("density")
        self.ax_density.set_title("Densities vs. time")
        self.ax_density.legend(loc="upper right", fontsize=9)

        self.anim = FuncAnimation(self.fig, self._on_frame, interval=50, cache_frame_data=False)

    # ---- callbacks --------------------------------------------------------
    def _on_live_param_change(self, _val):
        m = self.model
        m.b1, m.b2 = self.s_b1.val, self.s_b2.val
        m.d1, m.d2 = self.s_d1.val, self.s_d2.val
        m.gamma = self.s_gamma.val
        m.dt = self.s_dt.val
        self.steps_per_frame = int(self.s_speed.val)

    def _on_neighborhood_change(self, label):
        self.model.neighborhood = "N1" if "8" in label else "N0"

    def _on_play_pause(self, _event):
        self.running = not self.running
        self.btn_play.label.set_text("Pause" if self.running else "Play")

    def _on_reset(self, _event):
        m = self.model
        m.p1_init, m.p2_init = self.s_p1.val, self.s_p2.val
        m.size = int(self.s_size.val)
        m.reset()

        # grid size may have changed shape -> recreate the image artist
        self.im.remove()
        cmap = ListedColormap(["white", "red", "blue"])
        self.im = self.ax_lattice.imshow(m.grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        self.ax_lattice.set_xticks([])
        self.ax_lattice.set_yticks([])
        self.ax_lattice.set_title("t = 0.00")

        self.line1.set_data([], [])
        self.line2.set_data([], [])
        self.ax_density.set_xlim(0, 50)
        self.fig.canvas.draw_idle()

    def _on_single_step(self, _event):
        was_running = self.running
        self.running = False
        self.model.step()
        self._redraw()
        self.running = was_running

    def _on_frame(self, _frame):
        if self.running:
            for _ in range(self.steps_per_frame):
                self.model.step()
            self._redraw()
        return (self.im, self.line1, self.line2)

    def _redraw(self):
        m = self.model
        self.im.set_data(m.grid)
        self.ax_lattice.set_title(f"t = {m.time:.2f}")

        self.line1.set_data(m.history_t, m.history_f1)
        self.line2.set_data(m.history_t, m.history_f2)
        if m.time > self.ax_density.get_xlim()[1]:
            self.ax_density.set_xlim(0, m.time * 1.3)

        f1, f2 = m.history_f1[-1], m.history_f2[-1]
        f0 = 1.0 - f1 - f2
        self.info_text.set_text(
            f"density: producers={f1:.3f}  sensitive={f2:.3f}  vacant={f0:.3f}"
        )
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == "__main__":
    app = ColicinApp()
    app.show()