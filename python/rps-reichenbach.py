"""
Interactive lattice simulation of the stochastic spatial rock-paper-scissors
model with mobility, from:

    Reichenbach, T., Mobilia, M. & Frey, E. (2007). Mobility promotes and
    jeopardizes biodiversity in rock-paper-scissors games. Nature 448,
    1046-1049.

MODEL (Methods summary of the paper)
-------------------------------------
Individuals of three cyclically-dominant species A, B, C (a stochastic
lattice version of the May & Leonard 1975 model, following Durrett & Levin)
live on an L x L square lattice with periodic boundary conditions. Each
site is either empty (0) or occupied by one individual of species
A (1), B (2) or C (3), with cyclic dominance A -> B -> C -> A (A beats B,
B beats C, C beats A -- "rock crushes scissors, scissors cut paper, paper
wraps rock").

Two neighboring sites can undergo three possible elementary reactions,
each a Poisson process:

    Selection (rate sigma):  predator, prey        -> predator, empty
                              (the dominated species is killed)
    Reproduction (rate mu):  individual, empty      -> individual, individual
                              (only onto empty sites -> finite carrying capacity)
    Exchange (rate eps):     X, Y                   -> Y, X
                              (any two neighbors -- including an empty site --
                               swap places; this is what gives individuals
                               their mobility/diffusion)

The paper shows that the quantity M = 2*eps/N (N = number of lattice sites)
-- the "mobility", i.e. roughly the fraction of the lattice area an
individual explores per unit time -- controls the fate of the system:
below a critical value Mc all three species coexist indefinitely and
self-organize into rotating spiral waves; above Mc the spirals outgrow the
lattice and biodiversity collapses to a single species. For sigma = mu = 1
the paper reports Mc = (4.5 +/- 0.5) x 10^-4.

SIMULATION SCHEME
------------------
The paper's own microscopic definition is: a random individual is chosen,
paired with a random nearest neighbor, and (using the Gillespie algorithm)
one of the possible reactions on that bond fires according to its rate.
This script implements a numerically equivalent, fully vectorized version
using the same "rate * dt = probability in a short interval dt" convention
used for the colicin model. Every elementary "sweep" splits the lattice's
bonds into 4 non-overlapping sets (even/odd horizontal bonds, even/odd
vertical bonds -- a checkerboard bond decomposition), so that every site
participates in exactly one interaction with each of its 4 neighbors per
sweep, with no two simultaneously-updated bonds ever sharing a site. Within
each bond, the possible reactions (selection or exchange for two different
occupied neighbors; reproduction or exchange next to an empty site) compete
via probabilities rate * dt, exactly like the colicin script.

Requirements: numpy, matplotlib
    pip install numpy matplotlib

Run:
    python rps_mobility_lattice_model.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

EMPTY, A, B, C = 0, 1, 2, 3
# cyclic dominance: species i "beats" BEATS[i]  (A beats B, B beats C, C beats A)
BEATS = np.array([0, B, C, A])  # index 0 (EMPTY) unused


def _pair_indices(L, offset):
    """Indices of the two non-overlapping sets of sites forming L/2 bonds
    along one lattice direction (periodic). offset=0 pairs (0,1),(2,3),...
    offset=1 pairs (1,2),(3,4),...,(L-1,0) -- the wrap-around bond."""
    if offset == 0:
        idx_a = np.arange(0, L, 2)
        idx_b = np.arange(1, L, 2)
    else:
        idx_a = np.concatenate([np.arange(1, L - 1, 2), [L - 1]])
        idx_b = np.concatenate([np.arange(2, L, 2), [0]])
    return idx_a, idx_b


def _normalize_pair(p1, p2):
    """Rescale two competing event probabilities so they sum to at most 1,
    preserving their relative odds. A no-op when p1+p2 <= 1 (the normal
    regime); only kicks in as a safety net when dt is too large for the
    linear rate*dt approximation to hold on its own -- e.g. a large
    epsilon (mobility) combined with the default dt."""
    total = p1 + p2
    if total > 1.0:
        return p1 / total, p2 / total
    return p1, p2


def _react_bond(a, b, sigma, mu, eps, dt, rng):
    """Apply the competing Poisson reactions (selection / reproduction /
    exchange) to a batch of independent bonds at once. `a` and `b` are
    same-shape integer arrays holding the states at the two ends of each
    bond. Returns the updated (a_new, b_new)."""
    a_new = a.copy()
    b_new = b.copy()
    r = rng.random(a.shape)

    p_repro = mu * dt
    p_exch = eps * dt
    p_sel = sigma * dt

    # each site pair only ever competes between two outcomes: reproduction
    # vs. exchange (if one side is empty), or selection vs. exchange (if
    # both sides are occupied by different species) -- normalize each pair
    # independently so probabilities never exceed 1.
    p_repro, p_exch_re = _normalize_pair(p_repro, p_exch)
    p_sel, p_exch_se = _normalize_pair(p_sel, p_exch)

    a_empty = (a == EMPTY) & (b != EMPTY)
    b_empty = (b == EMPTY) & (a != EMPTY)
    diff_species = (a != b) & (a != EMPTY) & (b != EMPTY)

    thr_re = p_repro
    thr_re_ex = p_repro + p_exch_re

    # --- a is empty, b holds an individual: reproduction or hop ---
    m = a_empty
    do_repro = m & (r < thr_re)
    do_exch = m & (r >= thr_re) & (r < thr_re_ex)
    a_new[do_repro] = b[do_repro]            # empty site gets a copy
    a_new[do_exch] = b[do_exch]              # individual hops into empty site
    b_new[do_exch] = EMPTY

    # --- b is empty, a holds an individual: reproduction or hop ---
    m = b_empty
    do_repro = m & (r < thr_re)
    do_exch = m & (r >= thr_re) & (r < thr_re_ex)
    b_new[do_repro] = a[do_repro]
    b_new[do_exch] = a[do_exch]
    a_new[do_exch] = EMPTY

    # --- both occupied by different species: selection or exchange ---
    thr_se = p_sel
    thr_se_ex = p_sel + p_exch_se

    a_wins = diff_species & (BEATS[a] == b)
    m = a_wins
    do_sel = m & (r < thr_se)
    do_exch = m & (r >= thr_se) & (r < thr_se_ex)
    b_new[do_sel] = EMPTY                    # b (the loser) dies
    a_side = a[do_exch].copy()
    a_new[do_exch] = b[do_exch]
    b_new[do_exch] = a_side

    b_wins = diff_species & (BEATS[b] == a)
    m = b_wins
    do_sel = m & (r < thr_se)
    do_exch = m & (r >= thr_se) & (r < thr_se_ex)
    a_new[do_sel] = EMPTY                    # a (the loser) dies
    a_side2 = a[do_exch].copy()
    a_new[do_exch] = b[do_exch]
    b_new[do_exch] = a_side2

    return a_new, b_new


class RPSLatticeModel:
    def __init__(self, size=100, sigma=1.0, mu=1.0, epsilon=1.0,
                 vacancy_init=0.25, dt=0.02, seed=None):
        self.size = int(size) + (int(size) % 2)  # force even grid size
        self.sigma, self.mu, self.epsilon = float(sigma), float(mu), float(epsilon)
        self.vacancy_init = float(vacancy_init)
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)

        self.time = 0.0
        self.history_t, self.history_fA, self.history_fB, self.history_fC = [], [], [], []
        self.grid = None
        self.reset()

    def reset(self):
        L = self.size
        p_each = max(0.0, (1.0 - self.vacancy_init) / 3.0)
        r = self.rng.random((L, L))
        grid = np.zeros((L, L), dtype=np.int8)
        grid[r < p_each] = A
        grid[(r >= p_each) & (r < 2 * p_each)] = B
        grid[(r >= 2 * p_each) & (r < 3 * p_each)] = C
        self.grid = grid
        self.time = 0.0
        self.history_t, self.history_fA, self.history_fB, self.history_fC = [], [], [], []
        self._record()

    @property
    def mobility(self):
        """M = 2*eps/N as defined in the paper (N = number of lattice sites)."""
        return 2.0 * self.epsilon / (self.size * self.size)

    def _apply_horizontal(self, offset):
        L = self.size
        idx_a, idx_b = _pair_indices(L, offset)
        a, b = self.grid[:, idx_a], self.grid[:, idx_b]
        a_new, b_new = _react_bond(a, b, self.sigma, self.mu, self.epsilon, self.dt, self.rng)
        self.grid[:, idx_a] = a_new
        self.grid[:, idx_b] = b_new

    def _apply_vertical(self, offset):
        L = self.size
        idx_a, idx_b = _pair_indices(L, offset)
        a, b = self.grid[idx_a, :], self.grid[idx_b, :]
        a_new, b_new = _react_bond(a, b, self.sigma, self.mu, self.epsilon, self.dt, self.rng)
        self.grid[idx_a, :] = a_new
        self.grid[idx_b, :] = b_new

    def sweep(self):
        """One elementary time step dt, covering every one of the 4
        nearest-neighbor bonds of every site exactly once, via 4
        non-overlapping (hence conflict-free) passes in random order."""
        passes = [lambda: self._apply_horizontal(0),
                  lambda: self._apply_horizontal(1),
                  lambda: self._apply_vertical(0),
                  lambda: self._apply_vertical(1)]
        for i in self.rng.permutation(4):
            passes[i]()
        self.time += self.dt
        self._record()

    def _record(self):
        n = self.size * self.size
        self.history_t.append(self.time)
        self.history_fA.append(np.count_nonzero(self.grid == A) / n)
        self.history_fB.append(np.count_nonzero(self.grid == B) / n)
        self.history_fC.append(np.count_nonzero(self.grid == C) / n)


# --------------------------------------------------------------------------
# Interactive matplotlib application
# --------------------------------------------------------------------------
class RPSApp:
    MC_REFERENCE = 4.5e-4  # critical mobility reported for sigma = mu = 1

    def __init__(self):
        self.model = RPSLatticeModel()
        self.running = False
        self.sweeps_per_frame = 2

        self.fig = plt.figure(figsize=(13, 8))
        self.fig.suptitle(
            "Reichenbach, Mobilia & Frey (2007): mobility in a spatial rock-paper-scissors game    "
            "RED = A   BLUE = B   GOLD = C   (A beats B beats C beats A)   black = vacant",
            fontsize=10,
        )

        # --- left column: all parameter-scan controls -------------------------
        # col_x leaves room for slider labels (drawn to the left of the bar)
        # so long ones like "epsilon (exchange/mobility)" don't clip at the
        # figure edge.
        col_x, col_w = 0.18, 0.21

        # --- sliders (single stacked column) ------------------------------
        row_y = [0.90 - i * 0.038 for i in range(7)]

        def add_slider(y, label, vmin, vmax, vinit, step=None):
            ax = self.fig.add_axes([col_x, y, col_w, 0.02])
            return Slider(ax, label, vmin, vmax, valinit=vinit, valstep=step)

        self.s_sigma = add_slider(row_y[0], "sigma (selection)", 0.0, 4.0, self.model.sigma)
        self.s_mu = add_slider(row_y[1], "mu (reproduction)", 0.0, 4.0, self.model.mu)
        self.s_eps = add_slider(row_y[2], "epsilon (exchange/mobility)", 0.0, 30.0, self.model.epsilon)
        self.s_dt = add_slider(row_y[3], "dt (time step)", 0.001, 0.05, self.model.dt)
        self.s_vac = add_slider(row_y[4], "initial vacancy fraction", 0.0, 0.9, self.model.vacancy_init)
        self.s_size = add_slider(row_y[5], "grid size (on reset)", 20, 300, self.model.size, step=2)
        self.s_speed = add_slider(row_y[6], "sweeps / frame", 1, 20, self.sweeps_per_frame, step=1)

        for s in (self.s_sigma, self.s_mu, self.s_eps, self.s_dt, self.s_speed):
            s.on_changed(self._on_live_param_change)

        # --- buttons ---------------------------------------------------------
        # Centered as a block: horizontally on the sliders' true visual span
        # (their labels are drawn to the *left* of col_x, so the block's
        # center isn't col_x + col_w/2 but shifted left of it), and
        # vertically on the density-vs-time plot next to it.
        block_x, block_w = 0.104, col_w

        self.ax_play = self.fig.add_axes([block_x, 0.2925, block_w, 0.045])
        self.btn_play = Button(self.ax_play, "Play")
        self.btn_play.on_clicked(self._on_play_pause)

        self.ax_reset = self.fig.add_axes([block_x, 0.2375, block_w, 0.045])
        self.btn_reset = Button(self.ax_reset, "Reset")
        self.btn_reset.on_clicked(self._on_reset)

        self.ax_step = self.fig.add_axes([block_x, 0.1825, block_w, 0.045])
        self.btn_step = Button(self.ax_step, "Single sweep")
        self.btn_step.on_clicked(self._on_single_step)

        # fontsize kept small so the (fairly long) mobility diagnostic line
        # stays within the left column and doesn't bleed into the density plot.
        self.info_text = self.fig.text(block_x, 0.13, "", fontsize=7)
        self.status_text = self.fig.text(block_x, 0.09, "", fontsize=8, color="darkred", weight="bold")

        # --- right column: the two graphical widgets, stacked -------------------
        self.ax_lattice = self.fig.add_axes([0.46, 0.52, 0.48, 0.40])
        cmap = ListedColormap(["black", "red", "blue", "gold"])
        self.im = self.ax_lattice.imshow(self.model.grid, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        self.ax_lattice.set_xticks([])
        self.ax_lattice.set_yticks([])
        self.ax_lattice.set_title("t = 0.00")

        self.ax_density = self.fig.add_axes([0.46, 0.06, 0.48, 0.40])
        (self.lineA,) = self.ax_density.plot([], [], color="red", lw=1.3, label="A")
        (self.lineB,) = self.ax_density.plot([], [], color="blue", lw=1.3, label="B")
        (self.lineC,) = self.ax_density.plot([], [], color="goldenrod", lw=1.3, label="C")
        self.ax_density.set_xlim(0, 50)
        self.ax_density.set_ylim(0, 1)
        self.ax_density.set_xlabel("time (generations)")
        self.ax_density.set_ylabel("density")
        self.ax_density.set_title("Species densities vs. time")
        self.ax_density.legend(loc="upper right", fontsize=9)

        self.anim = FuncAnimation(self.fig, self._on_frame, interval=50, cache_frame_data=False)

    def _on_live_param_change(self, _val):
        m = self.model
        m.sigma, m.mu, m.epsilon = self.s_sigma.val, self.s_mu.val, self.s_eps.val
        m.dt = self.s_dt.val
        self.sweeps_per_frame = int(self.s_speed.val)

    def _on_play_pause(self, _event):
        self.running = not self.running
        self.btn_play.label.set_text("Pause" if self.running else "Play")

    def _on_reset(self, _event):
        m = self.model
        m.vacancy_init = self.s_vac.val
        m.size = int(self.s_size.val)
        m.reset()

        self.im.remove()
        cmap = ListedColormap(["black", "red", "blue", "gold"])
        self.im = self.ax_lattice.imshow(m.grid, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        self.ax_lattice.set_xticks([])
        self.ax_lattice.set_yticks([])
        self.ax_lattice.set_title("t = 0.00")

        self.lineA.set_data([], [])
        self.lineB.set_data([], [])
        self.lineC.set_data([], [])
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
        return (self.im, self.lineA, self.lineB, self.lineC)

    def _redraw(self):
        m = self.model
        self.im.set_data(m.grid)
        self.ax_lattice.set_title(f"t = {m.time:.2f}")

        self.lineA.set_data(m.history_t, m.history_fA)
        self.lineB.set_data(m.history_t, m.history_fB)
        self.lineC.set_data(m.history_t, m.history_fC)
        if m.time > self.ax_density.get_xlim()[1]:
            self.ax_density.set_xlim(0, m.time * 1.3)

        fA, fB, fC = m.history_fA[-1], m.history_fB[-1], m.history_fC[-1]
        f0 = 1.0 - fA - fB - fC
        M = m.mobility
        relation = "<" if M < self.MC_REFERENCE else ">"
        self.info_text.set_text(
            f"density: A={fA:.3f}  B={fB:.3f}  C={fC:.3f}  vacant={f0:.3f}\n"
            f"mobility M = 2*eps/N = {M:.2e}   {relation} Mc~{self.MC_REFERENCE:.1e} "
            f"(reference value for sigma=mu=1)"
        )

        n_extinct = sum(f < 1e-6 for f in (fA, fB, fC))
        if n_extinct >= 2:
            survivor = ["A", "B", "C"][int(np.argmax([fA, fB, fC]))]
            self.status_text.set_text(f"Biodiversity LOST -- only species {survivor} remains")
        elif n_extinct == 1:
            dying = ["A", "B", "C"][int(np.argmin([fA, fB, fC]))]
            self.status_text.set_text(f"Species {dying} approaching extinction...")
        else:
            self.status_text.set_text("")

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == "__main__":
    app = RPSApp()
    app.show()