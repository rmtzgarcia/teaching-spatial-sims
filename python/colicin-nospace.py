"""
Interactive phase-portrait / time-series explorer for the "complete mixing"
(non-spatial, mean-field) model of colicin allelopathy from:

    Iwasa, Y., Nakamaru, M. & Levin, S.A. (1998)
    "Allelopathy of bacteria in a lattice population: competition between
    colicin-sensitive and colicin-producing strains."
    Evolutionary Ecology, 12, 785-802.

The non-spatial (mean-field) dynamical system is Eq. (4a)-(4b) of the paper:

    d(rho1)/dt = beta1 * rho1 * (1 - rho1 - rho2) - delta1 * rho1
    d(rho2)/dt = beta2 * rho2 * (1 - rho1 - rho2) - delta2 * rho2 - gamma * rho1 * rho2

where
    rho1 = fraction of sites occupied by the colicin-PRODUCING strain (type 1, "C")
    rho2 = fraction of sites occupied by the colicin-SENSITIVE strain (type 2, "S")
    beta_i = intrinsic birth rate of strain i
    delta_i = intrinsic death rate of strain i
    gamma  = effectiveness of colicin (extra mortality it inflicts on S neighbours)

Following the paper's convention, beta1 = beta2 - c, where c is the "cost" of
colicin production (beta1 is held fixed here; dragging the beta2 slider moves
you along the paper's cost axis c = beta2 - beta1). The gamma slider moves you
along the paper's "effectiveness of colicin" axis (Fig. 1A).

Two widgets are shown side by side:
  LEFT  - phase portrait (rho1 vs rho2): nullclines, fixed points (stability
          shown by marker style), the background velocity field, and a family
          of rho1/rho2 trajectories from representative initial conditions.
  RIGHT - population (rho1, rho2) vs time for those same trajectories.

Two sliders let you manipulate beta2 (cost of colicin production, via
c = beta2 - beta1) and gamma (colicin effectiveness) to show the two possible
regimes of the model:
  * Monostable dominance of S  - the colicin-sensitive strain always wins,
    regardless of starting frequency.
  * Bistability                - either strain wins depending on which one
    starts more abundant; an unstable interior saddle point separates the two
    basins of attraction.

Click anywhere inside the simplex (rho1 + rho2 < 1) in the phase-portrait
panel to add your own trajectory from that starting point; use the "Reset
trajectories" button to go back to the default set.

Requires: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------------
# Fixed background parameters (only beta2 and gamma are exposed as sliders,
# as requested). beta1 is held fixed so that beta2 directly controls the
# paper's cost of colicin production, c = beta2 - beta1.
# ----------------------------------------------------------------------
BETA1 = 3.0      # intrinsic birth rate of the colicin-producing strain (C)
DELTA1 = 1.0     # death rate of C
DELTA2 = 1.0     # death rate of S

BETA2_INIT = 3.0
GAMMA_INIT = 2.0

T_MAX = 40.0     # integration horizon for trajectories
N_T = 400        # number of time points sampled for plotting

COLOR_C = "#C0392B"   # colicin-producing strain (rho1) - red
COLOR_S = "#3B6EA5"   # colicin-sensitive strain (rho2) - blue
TRAJ_CMAP = plt.get_cmap("tab10")

# The app starts with no trajectories plotted; click inside the phase
# portrait to add one. DEFAULT_ICS is kept only as the set restored by the
# "Reset trajectories" button... actually reset just clears back to empty,
# see on_reset().
DEFAULT_ICS = []
MAX_CUSTOM_TRAJECTORIES = 8


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def rhs(t, y, beta1, beta2, delta1, delta2, gamma):
    rho1, rho2 = y
    rho1 = max(rho1, 0.0)
    rho2 = max(rho2, 0.0)
    drho1 = beta1 * rho1 * (1.0 - rho1 - rho2) - delta1 * rho1
    drho2 = beta2 * rho2 * (1.0 - rho1 - rho2) - delta2 * rho2 - gamma * rho1 * rho2
    return [drho1, drho2]


def jacobian(rho1, rho2, beta1, beta2, delta1, delta2, gamma):
    df1_drho1 = beta1 * (1.0 - 2.0 * rho1 - rho2) - delta1
    df1_drho2 = -beta1 * rho1
    df2_drho1 = -(beta2 + gamma) * rho2
    df2_drho2 = beta2 * (1.0 - rho1 - 2.0 * rho2) - delta2 - gamma * rho1
    return np.array([[df1_drho1, df1_drho2], [df2_drho1, df2_drho2]])


def fixed_points(beta1, beta2, delta1, delta2, gamma):
    """Return dict of candidate fixed points (only those with non-negative
    coordinates are kept, except the origin which always exists)."""
    K1 = 1.0 - delta1 / beta1   # C-only equilibrium density
    K2 = 1.0 - delta2 / beta2   # S-only equilibrium density

    pts = {"origin": (0.0, 0.0)}
    if K1 > 0:
        pts["C_only"] = (K1, 0.0)
    if K2 > 0:
        pts["S_only"] = (0.0, K2)

    # Interior intersection of the two non-trivial nullclines:
    #   rho1-nullcline: rho1 + rho2 = K1
    #   rho2-nullcline: rho2 = K2 - (1 + gamma/beta2) * rho1
    if gamma > 1e-9 and K1 > 0 and K2 > 0:
        rho1_i = (K2 - K1) * beta2 / gamma
        rho2_i = K1 - rho1_i
        if rho1_i > 1e-6 and rho2_i > 1e-6:
            pts["interior"] = (rho1_i, rho2_i)

    return pts, K1, K2


def stability(rho1, rho2, beta1, beta2, delta1, delta2, gamma):
    J = jacobian(rho1, rho2, beta1, beta2, delta1, delta2, gamma)
    eig = np.linalg.eigvals(J)
    real = eig.real
    if np.all(real < -1e-9):
        return "stable"
    elif np.all(real > 1e-9):
        return "unstable"
    else:
        return "saddle"


def classify_regime(beta1, beta2, delta1, delta2, gamma):
    pts, K1, K2 = fixed_points(beta1, beta2, delta1, delta2, gamma)
    c_stable = "C_only" in pts and stability(*pts["C_only"], beta1, beta2, delta1, delta2, gamma) == "stable"
    s_stable = "S_only" in pts and stability(*pts["S_only"], beta1, beta2, delta1, delta2, gamma) == "stable"

    if c_stable and s_stable:
        label = "BISTABLE  (outcome depends on initial frequencies)"
    elif s_stable and not c_stable:
        label = "MONOSTABLE  —  colicin-sensitive strain (S) always wins"
    elif c_stable and not s_stable:
        label = "MONOSTABLE  —  colicin-producing strain (C) always wins"
    else:
        label = "no locally stable single-strain equilibrium"
    return label, pts, K1, K2


# ----------------------------------------------------------------------
# Trajectory integration
# ----------------------------------------------------------------------
def integrate_trajectory(ic, beta1, beta2, delta1, delta2, gamma, t_max=T_MAX, n_t=N_T):
    sol = solve_ivp(
        rhs, [0.0, t_max], ic,
        args=(beta1, beta2, delta1, delta2, gamma),
        dense_output=True, max_step=0.05, rtol=1e-7, atol=1e-9,
    )
    t = np.linspace(0.0, t_max, n_t)
    y = sol.sol(t)
    return t, np.clip(y[0], 0, None), np.clip(y[1], 0, None)


# ----------------------------------------------------------------------
# GUI application
# ----------------------------------------------------------------------
class ColicinPhasePortraitApp:
    def __init__(self):
        self.beta1 = BETA1
        self.delta1 = DELTA1
        self.delta2 = DELTA2
        self.beta2 = BETA2_INIT
        self.gamma = GAMMA_INIT
        self.custom_ics = []  # user-added initial conditions (from mouse clicks)

        self.fig = plt.figure(figsize=(13, 6.6))
        self.fig.suptitle(
            "Colicin allelopathy — complete-mixing (non-spatial) model  "
            "(Iwasa, Nakamaru & Levin 1998, Eq. 4)",
            fontsize=13,
        )

        gs = self.fig.add_gridspec(
            2, 2, width_ratios=[1, 1], height_ratios=[20, 1],
            left=0.07, right=0.97, top=0.88, bottom=0.24, wspace=0.28,
        )
        self.ax_phase = self.fig.add_subplot(gs[0, 0])
        self.ax_time = self.fig.add_subplot(gs[0, 1])

        # --- sliders ---
        self.fig.text(0.20, 0.135, r"$\beta_2$   (S birth rate; cost $c=\beta_2-\beta_1$)",
                       fontsize=10, ha="left")
        self.fig.text(0.20, 0.075, r"$\gamma$   (colicin effectiveness)",
                       fontsize=10, ha="left")
        ax_beta2 = self.fig.add_axes([0.20, 0.105, 0.34, 0.03])
        ax_gamma = self.fig.add_axes([0.20, 0.045, 0.34, 0.03])
        self.slider_beta2 = Slider(
            ax_beta2, "", 3.0, 10.0, valinit=self.beta2, valstep=0.05, color=COLOR_S,
        )
        self.slider_gamma = Slider(
            ax_gamma, "", 0.0, 20.0, valinit=self.gamma, valstep=0.1, color=COLOR_C,
        )
        self.slider_beta2.on_changed(self.on_slider_change)
        self.slider_gamma.on_changed(self.on_slider_change)

        # --- reset button ---
        ax_reset = self.fig.add_axes([0.62, 0.065, 0.15, 0.05])
        self.btn_reset = Button(ax_reset, "Reset trajectories")
        self.btn_reset.on_clicked(self.on_reset)

        # --- fixed-parameter readout ---
        self.fig.text(
            0.80, 0.115,
            r"Fixed: $\beta_1=%.1f,\ \delta_1=\delta_2=%.1f$" % (self.beta1, self.delta1),
            fontsize=10,
        )
        self.fig.text(
            0.80, 0.055,
            "Click inside the phase portrait\nto launch a new trajectory.",
            fontsize=9, color="dimgray",
        )

        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        self.redraw()

    # ------------------------------------------------------------
    def current_ics(self):
        return DEFAULT_ICS + self.custom_ics

    def on_slider_change(self, _val):
        self.beta2 = self.slider_beta2.val
        self.gamma = self.slider_gamma.val
        self.redraw()

    def on_reset(self, _event):
        self.custom_ics = []
        self.redraw()

    def on_click(self, event):
        if event.inaxes is not self.ax_phase:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        if x < 0 or y < 0 or x + y >= 0.995:
            return  # outside the biologically valid simplex
        self.custom_ics.append((x, y))
        if len(self.custom_ics) > MAX_CUSTOM_TRAJECTORIES:
            self.custom_ics.pop(0)
        self.redraw()

    # ------------------------------------------------------------
    def redraw(self):
        beta1, beta2 = self.beta1, self.beta2
        delta1, delta2, gamma = self.delta1, self.delta2, self.gamma

        regime_label, pts, K1, K2 = classify_regime(beta1, beta2, delta1, delta2, gamma)
        c_cost = beta2 - beta1

        ax = self.ax_phase
        ax.cla()

        lim = max(1.02, K1 + 0.05, K2 + 0.05)
        lim = min(lim, 1.05)

        # ---- velocity field (background vector field) ----
        n_grid = 16
        gx = np.linspace(0.0, lim, n_grid)
        gy = np.linspace(0.0, lim, n_grid)
        GX, GY = np.meshgrid(gx, gy)
        U = np.full_like(GX, np.nan)
        V = np.full_like(GY, np.nan)
        for i in range(n_grid):
            for j in range(n_grid):
                r1, r2 = GX[i, j], GY[i, j]
                if r1 + r2 <= 1.0:
                    d1, d2 = rhs(0, [r1, r2], beta1, beta2, delta1, delta2, gamma)
                    U[i, j], V[i, j] = d1, d2
        speed = np.sqrt(U ** 2 + V ** 2)
        speed_safe = np.where(speed > 1e-9, speed, 1.0)
        ax.quiver(
            GX, GY, U / speed_safe, V / speed_safe, speed,
            cmap="Greys", alpha=0.45, scale=28, width=0.0035, pivot="mid",
        )

        # ---- simplex boundary (rho0 = 0 line) ----
        ax.plot([0, 1], [1, 0], color="gray", lw=1, ls=":", zorder=1)

        # ---- nullclines ----
        # rho1-nullclines: rho1 = 0 (the rho2 axis) and rho1+rho2 = K1
        ax.axvline(0, color=COLOR_C, lw=1.3, alpha=0.35, zorder=1)
        if K1 > 0:
            xs = np.linspace(0, K1, 50)
            ax.plot(xs, K1 - xs, color=COLOR_C, lw=2, ls="--",
                     label=r"$\dot\rho_1=0$ (non-trivial): $\rho_1+\rho_2=K_1$")
        # rho2-nullclines: rho2 = 0 (the rho1 axis) and rho2 = K2 - (1+gamma/beta2) rho1
        ax.axhline(0, color=COLOR_S, lw=1.3, alpha=0.35, zorder=1)
        if K2 > 0:
            slope = 1.0 + gamma / beta2
            x_max = min(lim, K2 / slope) if slope > 0 else lim
            xs = np.linspace(0, x_max, 50)
            ys = K2 - slope * xs
            ax.plot(xs, ys, color=COLOR_S, lw=2, ls="--",
                     label=r"$\dot\rho_2=0$ (non-trivial): $\rho_2=K_2-(1+\gamma/\beta_2)\rho_1$")

        # ---- trajectories ----
        ics = self.current_ics()
        traj_data = []
        for k, ic in enumerate(ics):
            color = TRAJ_CMAP(k % 10)
            t, r1, r2 = integrate_trajectory(list(ic), beta1, beta2, delta1, delta2, gamma)
            traj_data.append((t, r1, r2, color))
            ax.plot(r1, r2, color=color, lw=1.8, alpha=0.9, zorder=3)
            ax.plot(ic[0], ic[1], marker="o", ms=4.5, color=color, zorder=4)
            # direction arrow at ~65% of the path
            idx = int(0.65 * len(r1))
            if idx < len(r1) - 2:
                ax.annotate(
                    "", xy=(r1[idx + 2], r2[idx + 2]), xytext=(r1[idx], r2[idx]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6), zorder=4,
                )

        # ---- fixed points ----
        def plot_fp(name, marker_stable="o", marker_other="s"):
            if name not in pts:
                return
            x, y = pts[name]
            st = stability(x, y, beta1, beta2, delta1, delta2, gamma)
            if st == "stable":
                ax.plot(x, y, marker="o", ms=13, mfc="black", mec="white", mew=1.5, zorder=5)
            elif st == "unstable":
                ax.plot(x, y, marker="o", ms=13, mfc="white", mec="black", mew=2, zorder=5)
            else:  # saddle
                ax.plot(x, y, marker="X", ms=13, mfc="orange", mec="black", mew=1.2, zorder=5)

        for name in ("origin", "C_only", "S_only", "interior"):
            plot_fp(name)

        ax.set_xlim(-0.02, lim)
        ax.set_ylim(-0.02, lim)
        ax.set_xlabel(r"$\rho_1$  (colicin-producing strain, C)", color=COLOR_C)
        ax.set_ylabel(r"$\rho_2$  (colicin-sensitive strain, S)", color=COLOR_S)
        ax.set_title(
            r"Phase portrait   ($\beta_2=%.2f,\ \gamma=%.2f,\ c=\beta_2-\beta_1=%.2f$)"
            % (beta2, gamma, c_cost),
            fontsize=10.5,
        )
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w", mfc="black", mec="white", ms=10, label="stable node"),
            plt.Line2D([0], [0], marker="o", color="w", mfc="white", mec="black", ms=10, label="unstable node"),
            plt.Line2D([0], [0], marker="X", color="w", mfc="orange", mec="black", ms=10, label="saddle"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

        ax.text(
            0.02, -0.20, "Regime: " + regime_label,
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            color=("#8B1A1A" if "S always" not in regime_label and "BISTABLE" in regime_label else "black"),
        )

        # ---- time-series panel ----
        axt = self.ax_time
        axt.cla()
        for t, r1, r2, color in traj_data:
            axt.plot(t, r1, color=color, lw=1.8, ls="-")
            axt.plot(t, r2, color=color, lw=1.8, ls="--")
        axt.set_xlabel("time")
        axt.set_ylabel("population density")
        axt.set_ylim(-0.02, max(1.0, K1 + 0.05, K2 + 0.05))
        axt.set_title("Population dynamics for the trajectories at left", fontsize=10.5)
        line_handles = [
            plt.Line2D([0], [0], color=COLOR_C, lw=2, ls="-", label=r"$\rho_1$ (C, producer)"),
            plt.Line2D([0], [0], color=COLOR_S, lw=2, ls="--", label=r"$\rho_2$ (S, susceptible)"),
        ]
        axt.legend(handles=line_handles, loc="upper right", fontsize=9)
        axt.grid(alpha=0.25)

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def main():
    app = ColicinPhasePortraitApp()
    app.show()


if __name__ == "__main__":
    main()