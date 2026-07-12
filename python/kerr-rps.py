"""
Interactive lattice simulation of the C-S-R (colicinogenic / sensitive /
resistant) rock-paper-scissors model from:

    Kerr, B., Riley, M. A., Feldman, M. W. & Bohannan, B. J. M. (2002).
    Local dispersal promotes biodiversity in a real-life game of
    rock-paper-scissors. Nature 418, 171-174.

MODEL (Box 1, "Lattice-based simulation")
------------------------------------------
Three E. coli strains form a non-transitive (rock-paper-scissors) triplet:
    C = colicin-producing (kills S)                -> drawn RED
    S = colicin-sensitive (outgrows R)              -> drawn BLUE
    R = colicin-resistant (outgrows C)              -> drawn GREEN
so that C beats S, S beats R, and R beats C -- exactly as in the paper's
Figure 1 colour scheme (empty sites are white).

Individuals live on an L x L square lattice with periodic boundaries. Every
site is empty or occupied by one of C, S, R. The paper uses a simple
*asynchronous* probabilistic update: repeatedly pick a random focal lattice
point and change its state according to these rules, where f_i is the
fraction of the focal point's neighborhood occupied by strain i:

    - if the focal point is EMPTY: it becomes occupied by strain i with
      probability f_i (i.e. with probability equal to i's share of the
      neighborhood -- exactly as if the empty site copied the state of a
      uniformly random neighbor); it stays empty with the remaining
      probability (the neighborhood's empty fraction).
    - if the focal point holds a C, it dies (becomes empty) with a fixed
      probability D_C.
    - if the focal point holds an R, it dies with a fixed probability D_R.
    - if the focal point holds an S, it dies with probability
          D_S = D_S0 + tau * f_C
      i.e. S has a baseline death probability D_S0, plus extra mortality
      proportional to the local density of colicin-producing neighbors
      (tau = "toxicity" of C).

The paper's time unit is the "epoch": 62,500 (= 250x250) individual
asynchronous point updates, i.e. one attempted update per lattice site on
average. This script applies the exact same per-site probabilities, but
updates every site of the lattice *simultaneously* once per epoch (a
vectorized numpy sweep) -- the natural, efficient parallel counterpart of
"one attempted update per site per epoch" used for the previous scripts in
this series.

The KEY manipulation in the paper is the size of the neighborhood used to
compute f_C, f_S, f_R (and hence used for both reproduction into empty
sites and colicin exposure):

    - "local"  neighborhood = the 8 surrounding lattice points (Moore
      neighborhood) -> interaction and dispersal are spatially local.
    - "global" neighborhood = every other point on the lattice -> the
      community is effectively well-mixed ("mass action").

With the local neighborhood the three strains chase each other in patches
and coexist indefinitely (Fig. 1a-c); with the global neighborhood, S goes
extinct almost immediately and then R outcompetes C, leaving only R
(Fig. 1d). This GUI lets you flip between the two and watch the difference
directly, exactly as in the paper.

Requirements: numpy, matplotlib
    pip install numpy matplotlib

Run:
    python csr_lattice_model.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation

EMPTY, C, S, R = 0, 1, 2, 3

# the 8 surrounding points ("Moore" neighborhood) used for the local case
MOORE_SHIFTS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]


class CSRLatticeModel:
    def __init__(self, size=150, D_C=1 / 3, D_S0=1 / 4, D_R=10 / 32, tau=3 / 4,
                 vacancy_init=0.25, neighborhood="local", seed=None):
        self.size = int(size)
        self.D_C, self.D_S0, self.D_R, self.tau = float(D_C), float(D_S0), float(D_R), float(tau)
        self.vacancy_init = float(vacancy_init)
        self.neighborhood = neighborhood  # "local" or "global"
        self.rng = np.random.default_rng(seed)

        self.time = 0.0  # measured in epochs
        self.history_t, self.history_fC, self.history_fS, self.history_fR = [], [], [], []
        self.grid = None
        self.reset()

    def reset(self):
        L = self.size
        p_each = max(0.0, (1.0 - self.vacancy_init) / 3.0)
        r = self.rng.random((L, L))
        grid = np.zeros((L, L), dtype=np.int8)
        grid[r < p_each] = C
        grid[(r >= p_each) & (r < 2 * p_each)] = S
        grid[(r >= 2 * p_each) & (r < 3 * p_each)] = R
        self.grid = grid
        self.time = 0.0
        self.history_t, self.history_fC, self.history_fS, self.history_fR = [], [], [], []
        self._record()

    def _local_fraction(self, mask):
        total = np.zeros(self.grid.shape, dtype=float)
        for dx, dy in MOORE_SHIFTS:
            total += np.roll(np.roll(mask, dx, axis=0), dy, axis=1)
        return total / len(MOORE_SHIFTS)

    def _neighbor_fractions(self):
        isC = self.grid == C
        isS = self.grid == S
        isR = self.grid == R
        if self.neighborhood == "local":
            fC = self._local_fraction(isC)
            fS = self._local_fraction(isS)
            fR = self._local_fraction(isR)
        else:  # "global": every other point on the lattice (well-mixed)
            n = self.size * self.size
            countC, countS, countR = isC.sum(), isS.sum(), isR.sum()
            fC = (countC - isC) / (n - 1)
            fS = (countS - isS) / (n - 1)
            fR = (countR - isR) / (n - 1)
        return fC, fS, fR

    def sweep(self):
        """Advance the lattice by one epoch (every site updated once,
        synchronously, using the same probabilities as the paper's
        asynchronous single-site update rule)."""
        grid = self.grid
        isEmpty = grid == EMPTY
        isC = grid == C
        isS = grid == S
        isR = grid == R

        fC, fS, fR = self._neighbor_fractions()
        D_S = np.clip(self.D_S0 + self.tau * fC, 0.0, 1.0)

        r = self.rng.random(grid.shape)
        new_grid = grid.copy()

        # Empty sites: become C, S or R with probability = that strain's
        # local share of the neighborhood (else remain empty).
        thrC = fC
        thrS = fC + fS
        thrR = np.clip(fC + fS + fR, 0.0, 1.0)
        m = isEmpty
        new_grid[m & (r < thrC)] = C
        new_grid[m & (r >= thrC) & (r < thrS)] = S
        new_grid[m & (r >= thrS) & (r < thrR)] = R

        # Occupied sites: die (become empty) with their strain's death probability.
        new_grid[isC & (r < self.D_C)] = EMPTY
        new_grid[isR & (r < self.D_R)] = EMPTY
        new_grid[isS & (r < D_S)] = EMPTY

        self.grid = new_grid
        self.time += 1.0  # one epoch
        self._record()

    def _record(self):
        n = self.size * self.size
        self.history_t.append(self.time)
        self.history_fC.append(np.count_nonzero(self.grid == C) / n)
        self.history_fS.append(np.count_nonzero(self.grid == S) / n)
        self.history_fR.append(np.count_nonzero(self.grid == R) / n)

    def coexistence_constraint_ok(self):
        """The paper requires D_S0 < D_R < D_C < (D_S0+tau)/(1+tau) for the
        mixed system to guarantee S beats R, R beats C, C beats S."""
        upper = (self.D_S0 + self.tau) / (1.0 + self.tau)
        return self.D_S0 < self.D_R < self.D_C < upper


# --------------------------------------------------------------------------
# Interactive matplotlib application
# --------------------------------------------------------------------------
class CSRApp:
    def __init__(self):
        self.model = CSRLatticeModel()
        self.running = False
        self.sweeps_per_frame = 1

        self.fig = plt.figure(figsize=(13, 8))
        self.fig.suptitle(
            "Kerr, Riley, Feldman & Bohannan (2002): local vs. global dispersal in a C-S-R rock-paper-scissors community    "
            "RED = C (producer)   BLUE = S (sensitive)   GREEN = R (resistant)   white = vacant",
            fontsize=9.5,
        )

        # --- left column: all parameter-scan controls -------------------------
        # col_x leaves room for slider labels (drawn to the left of the bar)
        # so long ones like "D_S0 (sensitive baseline death)" don't clip at
        # the figure edge.
        col_x, col_w = 0.18, 0.21

        # --- sliders (single stacked column) ------------------------------
        row_y = [0.90 - i * 0.038 for i in range(7)]

        def add_slider(y, label, vmin, vmax, vinit, step=None):
            ax = self.fig.add_axes([col_x, y, col_w, 0.02])
            return Slider(ax, label, vmin, vmax, valinit=vinit, valstep=step)

        self.s_DC = add_slider(row_y[0], "D_C (producer death)", 0.0, 1.0, self.model.D_C)
        self.s_DS0 = add_slider(row_y[1], "D_S0 (sensitive baseline death)", 0.0, 1.0, self.model.D_S0)
        self.s_DR = add_slider(row_y[2], "D_R (resistant death)", 0.0, 1.0, self.model.D_R)
        self.s_tau = add_slider(row_y[3], "tau (colicin toxicity)", 0.0, 2.0, self.model.tau)
        self.s_vac = add_slider(row_y[4], "initial vacancy fraction", 0.0, 0.9, self.model.vacancy_init)
        self.s_size = add_slider(row_y[5], "grid size (on reset)", 20, 250, self.model.size, step=10)
        self.s_speed = add_slider(row_y[6], "epochs / frame", 1, 20, self.sweeps_per_frame, step=1)

        for s in (self.s_DC, self.s_DS0, self.s_DR, self.s_tau, self.s_speed):
            s.on_changed(self._on_live_param_change)

        # --- neighborhood choice + buttons ---------------------------------
        # Centered as a block: horizontally on the sliders' true visual span
        # (their labels are drawn to the *left* of col_x, so the block's
        # center isn't col_x + col_w/2 but shifted left of it), and
        # vertically on the density-vs-time plot next to it.
        block_x, block_w = 0.094, col_w

        self.ax_radio = self.fig.add_axes([block_x, 0.283, block_w, 0.14])
        self.ax_radio.set_title("Neighborhood", fontsize=9)
        self.radio = RadioButtons(self.ax_radio, ("local (8 neighbors)", "global (well-mixed)"), active=0)
        self.radio.on_clicked(self._on_neighborhood_change)

        self.ax_play = self.fig.add_axes([block_x, 0.208, block_w, 0.045])
        self.btn_play = Button(self.ax_play, "Play")
        self.btn_play.on_clicked(self._on_play_pause)

        self.ax_reset = self.fig.add_axes([block_x, 0.153, block_w, 0.045])
        self.btn_reset = Button(self.ax_reset, "Reset")
        self.btn_reset.on_clicked(self._on_reset)

        self.ax_step = self.fig.add_axes([block_x, 0.098, block_w, 0.045])
        self.btn_step = Button(self.ax_step, "Single epoch")
        self.btn_step.on_clicked(self._on_single_step)

        # fontsize kept small, and the constraint line shortened/wrapped, so
        # this (fairly long) diagnostic text stays within the left column
        # and doesn't bleed into the density plot.
        self.info_text = self.fig.text(block_x, 0.055, "", fontsize=7)
        self.status_text = self.fig.text(block_x, 0.02, "", fontsize=8, color="darkred", weight="bold")

        # --- right column: the two graphical widgets, stacked -------------------
        self.ax_lattice = self.fig.add_axes([0.46, 0.52, 0.48, 0.40])
        cmap = ListedColormap(["white", "red", "blue", "green"])
        self.im = self.ax_lattice.imshow(self.model.grid, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        self.ax_lattice.set_xticks([])
        self.ax_lattice.set_yticks([])
        self.ax_lattice.set_title("epoch = 0")

        self.ax_density = self.fig.add_axes([0.46, 0.06, 0.48, 0.40])
        (self.lineC,) = self.ax_density.plot([], [], color="red", lw=1.3, label="C")
        (self.lineS,) = self.ax_density.plot([], [], color="blue", lw=1.3, label="S")
        (self.lineR,) = self.ax_density.plot([], [], color="green", lw=1.3, label="R")
        self.ax_density.set_xlim(0, 50)
        self.ax_density.set_ylim(0, 1)
        self.ax_density.set_xlabel("time (epochs)")
        self.ax_density.set_ylabel("density")
        self.ax_density.set_title("Strain densities vs. time (cf. Fig. 1c/d)")
        self.ax_density.legend(loc="upper right", fontsize=9)

        self.anim = FuncAnimation(self.fig, self._on_frame, interval=50, cache_frame_data=False)

    def _on_live_param_change(self, _val):
        m = self.model
        m.D_C, m.D_S0, m.D_R, m.tau = self.s_DC.val, self.s_DS0.val, self.s_DR.val, self.s_tau.val
        self.sweeps_per_frame = int(self.s_speed.val)

    def _on_neighborhood_change(self, label):
        self.model.neighborhood = "local" if "local" in label else "global"

    def _on_play_pause(self, _event):
        self.running = not self.running
        self.btn_play.label.set_text("Pause" if self.running else "Play")

    def _on_reset(self, _event):
        m = self.model
        m.vacancy_init = self.s_vac.val
        m.size = int(self.s_size.val)
        m.reset()

        self.im.remove()
        cmap = ListedColormap(["white", "red", "blue", "green"])
        self.im = self.ax_lattice.imshow(m.grid, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        self.ax_lattice.set_xticks([])
        self.ax_lattice.set_yticks([])
        self.ax_lattice.set_title("epoch = 0")

        self.lineC.set_data([], [])
        self.lineS.set_data([], [])
        self.lineR.set_data([], [])
        self.ax_density.set_xlim(0, 50)
        self.status_text.set_text("")
        self.fig.canvas.draw_idle()

    def _on_single_step(self, _event):
        was_running = self.running
        self.running = False
        self.model.sweep()
        self._redraw()
        self.running = was_running

    def _on_frame(self, _frame):
        if self.running:
            for _ in range(self.sweeps_per_frame):
                self.model.sweep()
            self._redraw()
        return (self.im, self.lineC, self.lineS, self.lineR)

    def _redraw(self):
        m = self.model
        self.im.set_data(m.grid)
        self.ax_lattice.set_title(f"epoch = {m.time:.0f}   ({m.neighborhood})")

        self.lineC.set_data(m.history_t, m.history_fC)
        self.lineS.set_data(m.history_t, m.history_fS)
        self.lineR.set_data(m.history_t, m.history_fR)
        if m.time > self.ax_density.get_xlim()[1]:
            self.ax_density.set_xlim(0, m.time * 1.3)

        fC, fS, fR = m.history_fC[-1], m.history_fS[-1], m.history_fR[-1]
        f0 = 1.0 - fC - fS - fR
        ok = m.coexistence_constraint_ok()
        self.info_text.set_text(
            f"density: C={fC:.3f}  S={fS:.3f}  R={fR:.3f}  vacant={f0:.3f}\n"
            f"coexistence (D_S0<D_R<D_C<(D_S0+tau)/(1+tau)): "
            f"{'satisfied' if ok else 'VIOLATED'}"
        )

        n_extinct = sum(f < 1e-6 for f in (fC, fS, fR))
        if n_extinct >= 2:
            survivor = ["C", "S", "R"][int(np.argmax([fC, fS, fR]))]
            self.status_text.set_text(f"Biodiversity LOST -- only {survivor} remains")
        elif n_extinct == 1:
            dying = ["C", "S", "R"][int(np.argmin([fC, fS, fR]))]
            self.status_text.set_text(f"{dying} approaching extinction...")
        else:
            self.status_text.set_text("")

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == "__main__":
    app = CSRApp()
    app.show()