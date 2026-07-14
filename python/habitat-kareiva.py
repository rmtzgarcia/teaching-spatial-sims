#!/usr/bin/env python3
"""
Habitat Selection via Movement: a Markov-chain demo inspired by

    Kareiva, P. (1982). Experimental and Mathematical Analyses of Herbivore
    Movement: Quantifying the Influence of Plant Spacing and Quality on
    Foraging Discrimination. Ecological Monographs 52(3): 261-282.

Kareiva released marked flea beetles into a linear array (a "stepping-stone"
archipelago) of collard patches and asked: does a simple passive-diffusion
model explain where the beetles end up? In homogeneous arrays (all patches
equal quality), the answer was mostly yes. In heterogeneous arrays (patches
alternating "lush" and "stunted"), passive diffusion failed: beetles lingered
on good patches and hurried through bad ones. He fixed this with a general
continuous-time Markov chain in which each patch has its own emigration rate,
and showed that this simple rule -- movement rate depends on local patch
quality -- reproduces the observed "foraging discrimination" indices almost
exactly (his Fig. 11).

This app lets you build that same kind of linear array, assign each patch a
quality, and watch:

  1. a deterministic prediction of the population distribution over time,
     obtained by solving the master equation  dN/dt = N Q  (Q = the
     patch-to-patch intensity/generator matrix) via eigendecomposition --
     exactly Kareiva's Eq. 6 machinery, just without needing SciPy.
  2. an agent-based stochastic simulation (many independent "beetles"
     performing an exact continuous-time random walk, i.e. a Gillespie
     simulation) that should match the deterministic prediction, the same
     way Kareiva checked his Markov model's predictions against mark-
     recapture histograms.
  3. a "null" passive-diffusion reference (all patches given the same
     emigration rate, ignoring quality) plotted alongside the quality-aware
     prediction, so students can see exactly how much of the pattern is due
     to habitat selection rather than plain diffusion.

Run with:
    python3 habitat_selection_gui.py

Requires: numpy, matplotlib, tkinter. Self-contained, single file.
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ----------------------------------------------------------------------------
# Core model
#
# Patches are arranged on a line, indices 0..S-1, "distance" metres apart
# (Kareiva's linear arrays used 2, 3, 5, 6, 9, or 11 m spacing). Movement is
# restricted to nearest neighbours, exactly as Kareiva assumed for his
# 60-min mark-recapture windows ("very few beetles were observed to move
# beyond their neighbouring patches").
#
# Each patch i has a quality q_i in [0, 1] (0 = stunted, 1 = lush). The
# instantaneous emigration rate out of patch i is
#
#     rate_i = base_rate * (1 - alpha * q_i)      [quality-dependent model]
#     rate_i = base_rate                          [null / passive-diffusion]
#
# so that on a lush patch (q_i near 1) beetles linger longer (lower rate_i),
# reproducing Kareiva's empirical finding that flea beetles abandon lush
# patches less readily than stunted ones. When a beetle does leave patch i,
# it moves to one of its (1 or 2) neighbours with equal probability -- this
# is the same "stepping-stone" assumption Kareiva used.
#
# base_rate itself scales as mobility_k / distance, reflecting Kareiva's
# Fig. 5 finding that beetles move between patches faster when patches are
# closer together.
# ----------------------------------------------------------------------------

def build_generator(quality, mobility_k, distance, quality_effect=True, alpha=0.85):
    """
    Build the S x S continuous-time Markov chain generator (intensity) matrix
    Q for a linear array of patches. Q[i,j] (i != j) is the instantaneous
    rate of moving from patch i to patch j (nonzero only for neighbours);
    Q[i,i] = -sum_j Q[i,j].
    """
    quality = np.asarray(quality, dtype=float)
    S = len(quality)
    base_rate = mobility_k / max(distance, 1e-6)

    Q = np.zeros((S, S))
    for i in range(S):
        rate_i = base_rate * ((1 - alpha * quality[i]) if quality_effect else 1.0)
        rate_i = max(rate_i, 1e-6)
        neighbors = [j for j in (i - 1, i + 1) if 0 <= j < S]
        if not neighbors:
            continue
        share = rate_i / len(neighbors)
        for j in neighbors:
            Q[i, j] = share
        Q[i, i] = -rate_i
    return Q


def solve_master_equation(Q, N0, t_array):
    """
    Solve dN/dt = N Q (N a row vector of patch occupation probabilities)
    for each t in t_array, via eigendecomposition of Q^T (no SciPy needed):
    N(t)^T = V exp(Lambda t) V^-1 N(0)^T.
    """
    QT = Q.T
    eigvals, eigvecs = np.linalg.eig(QT)
    eigvecs_inv = np.linalg.inv(eigvecs)
    coeffs = eigvecs_inv @ N0
    out = []
    for t in t_array:
        Nt = eigvecs @ (np.exp(eigvals * t) * coeffs)
        out.append(np.real(Nt))
    return np.array(out)


def simulate_ctmc_walkers(Q, release_patch, n_walkers, t_max, seed=None):
    """
    Exact stochastic (Gillespie) simulation of n_walkers independent
    continuous-time random walkers on the patch network defined by Q,
    starting at release_patch. Returns a list of (times, patches) per
    walker giving its full jump history up to t_max.
    """
    rng = np.random.default_rng(seed)
    S = Q.shape[0]
    trajectories = []
    for _ in range(n_walkers):
        t = 0.0
        patch = release_patch
        times = [0.0]
        patches = [patch]
        while True:
            rate_out = -Q[patch, patch]
            if rate_out <= 1e-9:
                break
            dt = rng.exponential(1.0 / rate_out)
            t += dt
            if t >= t_max:
                break
            probs = Q[patch].copy()
            probs[patch] = 0.0
            probs = probs / probs.sum()
            patch = rng.choice(S, p=probs)
            times.append(t)
            patches.append(patch)
        trajectories.append((np.array(times), np.array(patches)))
    return trajectories


def positions_at(trajectories, t_query):
    """Look up each walker's patch at time t_query (step-function lookup)."""
    out = np.empty(len(trajectories), dtype=int)
    for k, (times, patches) in enumerate(trajectories):
        idx = np.searchsorted(times, t_query, side="right") - 1
        idx = max(idx, 0)
        out[k] = patches[idx]
    return out


def quality_preset(name, S):
    """Return a length-S quality array (0=stunted, 1=lush) for a preset name."""
    if name == "Homogeneous":
        return np.full(S, 0.5)
    if name == "Alternating (lush/stunted)":
        return np.array([1.0 if i % 2 == 0 else 0.0 for i in range(S)])
    if name == "Gradient (stunted -> lush)":
        return np.linspace(0.0, 1.0, S)
    if name == "One damaged patch (center)":
        q = np.full(S, 0.8)
        q[S // 2] = 0.05
        return q
    return np.full(S, 0.5)


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
# Small vertical slider used for per-patch quality in "Custom" mode
# ----------------------------------------------------------------------------
class VerticalQualitySlider(ttk.Frame):
    def __init__(self, parent, index, initial, command=None):
        super().__init__(parent)
        self._command = command
        self.index = index
        self.var = tk.DoubleVar(value=initial)
        ttk.Label(self, text=str(index)).pack()
        self.scale = ttk.Scale(self, from_=1.0, to=0.0, orient="vertical",
                               variable=self.var, command=self._on_change, length=90)
        self.scale.pack()
        self.value_label = ttk.Label(self, text=f"{initial:.2f}")
        self.value_label.pack()

    def _on_change(self, _evt=None):
        v = self.var.get()
        self.value_label.config(text=f"{v:.2f}")
        if self._command:
            self._command()

    def get(self):
        return self.var.get()

    def set(self, v):
        self.var.set(v)
        self.value_label.config(text=f"{v:.2f}")


# ----------------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------------
class HabitatSelectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Habitat Selection via Movement (after Kareiva 1982)")
        self.geometry("1250x760")

        self.traj_qual = None
        self.traj_null = None
        self.Q_qual = None
        self.Q_null = None
        self.N0 = None
        self.quality_sliders = []

        self._build_ui()
        self.resimulate()

    # -- UI construction ----------------------------------------------------
    def _build_ui(self):
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)

        controls = ttk.Frame(outer, padding=10)
        controls.pack(side="left", fill="y")

        ttk.Label(controls, text="Habitat selection via movement",
                 font=("", 12, "bold")).pack(anchor="w")
        ttk.Label(controls, text="(linear array, after Kareiva 1982)",
                 font=("", 9, "italic")).pack(anchor="w", pady=(0, 10))

        self.n_patches = LabeledSlider(controls, "Number of patches (S)", 3, 15, 9,
                                       resolution=1, is_int=True,
                                       command=self._on_structure_change)
        self.n_patches.pack(fill="x", pady=4)

        self.distance = LabeledSlider(controls, "Distance between patches (m)",
                                      1.0, 12.0, 3.0, resolution=0.5)
        self.distance.pack(fill="x", pady=4)

        self.mobility = LabeledSlider(controls, "Mobility constant  k", 0.1, 5.0, 1.0,
                                      resolution=0.1)
        self.mobility.pack(fill="x", pady=4)

        self.alpha = LabeledSlider(controls, "Quality effect strength  α", 0.0, 0.99, 0.85,
                                   resolution=0.01)
        self.alpha.pack(fill="x", pady=4)

        ttk.Label(controls, text="rate_i = k/distance × (1 - α·quality_i)",
                 font=("", 8), foreground="#555").pack(anchor="w", pady=(0, 8))

        ttk.Label(controls, text="Patch quality pattern:").pack(anchor="w")
        self.preset_var = tk.StringVar(value="Alternating (lush/stunted)")
        preset_menu = ttk.Combobox(controls, textvariable=self.preset_var, state="readonly",
                                   values=["Homogeneous", "Alternating (lush/stunted)",
                                           "Gradient (stunted -> lush)",
                                           "One damaged patch (center)", "Custom"])
        preset_menu.pack(fill="x", pady=(2, 4))
        preset_menu.bind("<<ComboboxSelected>>", lambda e: self._on_preset_change())

        self.release_patch = LabeledSlider(controls, "Release patch (index)", 0, 8, 4,
                                           resolution=1, is_int=True)
        self.release_patch.pack(fill="x", pady=4)

        self.n_walkers = LabeledSlider(controls, "Walkers per model", 50, 5000, 1000,
                                       resolution=50, is_int=True)
        self.n_walkers.pack(fill="x", pady=4)

        self.t_max = LabeledSlider(controls, "Max simulated time", 1, 200, 50,
                                   resolution=1)
        self.t_max.pack(fill="x", pady=4)

        ttk.Button(controls, text="Re-simulate", command=self.resimulate).pack(
            fill="x", pady=(10, 4))

        ttk.Separator(controls, orient="horizontal").pack(fill="x", pady=10)

        self.t_slider = LabeledSlider(controls, "Time shown", 0.1, 50, 50,
                                      resolution=0.1, command=self._on_time_change)
        self.t_slider.pack(fill="x", pady=4)

        self.play_var = tk.BooleanVar(value=False)
        self.play_btn = ttk.Button(controls, text="Play", command=self.toggle_play)
        self.play_btn.pack(fill="x", pady=(6, 4))

        self.show_null = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Show passive-diffusion null model",
                        variable=self.show_null, command=self.redraw).pack(
            anchor="w", pady=(8, 0))

        self.info_label = ttk.Label(controls, text="", justify="left", wraplength=250)
        self.info_label.pack(anchor="w", pady=(10, 0))

        # Custom quality slider strip (built/rebuilt on demand)
        self.custom_frame = ttk.LabelFrame(controls, text="Custom patch quality")
        self.custom_strip = ttk.Frame(self.custom_frame)
        self.custom_strip.pack()
        # not packed initially; shown only in Custom mode

        # Plot area
        plot_frame = ttk.Frame(outer)
        plot_frame.pack(side="right", fill="both", expand=True)

        self.fig = Figure(figsize=(8.5, 6.5), dpi=100)
        self.ax_quality = self.fig.add_subplot(211)
        self.ax_dist = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # -- Quality handling -----------------------------------------------------
    def _get_quality(self):
        S = self.n_patches.get()
        if self.preset_var.get() == "Custom":
            if len(self.quality_sliders) != S:
                self._rebuild_custom_sliders()
            return np.array([s.get() for s in self.quality_sliders])
        return quality_preset(self.preset_var.get(), S)

    def _rebuild_custom_sliders(self):
        for child in self.custom_strip.winfo_children():
            child.destroy()
        S = self.n_patches.get()
        base = quality_preset("Alternating (lush/stunted)", S)
        self.quality_sliders = []
        for i in range(S):
            vs = VerticalQualitySlider(self.custom_strip, i, base[i],
                                       command=self.redraw)
            vs.pack(side="left", padx=2)
            self.quality_sliders.append(vs)

    def _on_preset_change(self):
        if self.preset_var.get() == "Custom":
            self._rebuild_custom_sliders()
            self.custom_frame.pack(fill="x", pady=(8, 0))
        else:
            self.custom_frame.pack_forget()
        self.resimulate()

    def _on_structure_change(self, _v):
        S = self.n_patches.get()
        self.release_patch.scale.config(to=S - 1)
        if self.release_patch.get() > S - 1:
            self.release_patch.set(S - 1)
        if self.preset_var.get() == "Custom":
            self._rebuild_custom_sliders()
        self.resimulate()

    # -- Simulation -----------------------------------------------------------
    def resimulate(self):
        S = self.n_patches.get()
        quality = self._get_quality()
        mobility_k = self.mobility.get()
        distance = self.distance.get()
        alpha = self.alpha.get()
        release = min(self.release_patch.get(), S - 1)
        n_walkers = self.n_walkers.get()
        t_max = self.t_max.get()

        self.Q_qual = build_generator(quality, mobility_k, distance,
                                      quality_effect=True, alpha=alpha)
        self.Q_null = build_generator(quality, mobility_k, distance,
                                      quality_effect=False)

        self.N0 = np.zeros(S)
        self.N0[release] = 1.0

        self.traj_qual = simulate_ctmc_walkers(self.Q_qual, release, n_walkers, t_max, seed=None)
        self.traj_null = simulate_ctmc_walkers(self.Q_null, release, n_walkers, t_max, seed=None)

        self.t_slider.scale.config(from_=t_max / 100, to=t_max)
        self.t_slider.set(t_max)

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
        t_max = self.t_max.get()
        cur = self.t_slider.get()
        nxt = cur + max(t_max / 150, 0.05)
        if nxt > t_max:
            nxt = t_max / 100
        self.t_slider.set(nxt)
        self.redraw()
        self.after(40, self._animate_step)

    # -- Drawing ---------------------------------------------------------------
    def redraw(self):
        if self.Q_qual is None:
            return
        S = self.n_patches.get()
        quality = self._get_quality()
        t = self.t_slider.get()

        N_qual = solve_master_equation(self.Q_qual, self.N0, [t])[0]
        N_null = solve_master_equation(self.Q_null, self.N0, [t])[0]

        pos_qual = positions_at(self.traj_qual, t)
        counts_qual = np.bincount(pos_qual, minlength=S)
        frac_qual_emp = counts_qual / counts_qual.sum()

        x = np.arange(S)

        self.ax_quality.clear()
        colors = [(0.85 * (1 - q) + 0.1, 0.55 + 0.35 * q, 0.15 * (1 - q)) for q in quality]
        self.ax_quality.bar(x, quality, color=colors, edgecolor="k", linewidth=0.5)
        self.ax_quality.axvline(self.release_patch.get(), color="k", linestyle="--",
                                linewidth=1, alpha=0.6, label="release patch")
        self.ax_quality.set_ylim(0, 1.05)
        self.ax_quality.set_ylabel("quality")
        self.ax_quality.set_title(f"Patch quality (0 = stunted, 1 = lush), distance = "
                                  f"{self.distance.get():.1f} m between patches")
        self.ax_quality.legend(loc="upper right", fontsize=8)

        self.ax_dist.clear()
        width = 0.35
        self.ax_dist.bar(x - width / 2, frac_qual_emp, width=width, alpha=0.6,
                         color="tab:green", label="habitat-selection model (simulated beetles)")
        self.ax_dist.plot(x, N_qual, "o-", color="darkgreen", linewidth=2,
                          label="habitat-selection model (master equation)")

        if self.show_null.get():
            self.ax_dist.plot(x, N_null, "s--", color="tab:gray", linewidth=1.5,
                              label="passive-diffusion null (master equation)")

        self.ax_dist.set_xlabel("patch index")
        self.ax_dist.set_ylabel("fraction of population")
        self.ax_dist.set_title(f"Population distribution at t = {t:.2f}")
        self.ax_dist.legend(loc="upper right", fontsize=7)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw_idle()

        lush_idx = np.where(quality >= 0.5)[0]
        stunted_idx = np.where(quality < 0.5)[0]
        if len(lush_idx) and len(stunted_idx):
            disc_qual = N_qual[lush_idx].mean() / max(N_qual[stunted_idx].mean(), 1e-9)
            disc_null = N_null[lush_idx].mean() / max(N_null[stunted_idx].mean(), 1e-9)
            disc_text = (f"Discrimination index (mean lush / mean stunted occupancy):\n"
                        f"  habitat-selection model: {disc_qual:.2f}\n"
                        f"  passive-diffusion null:  {disc_null:.2f}")
        else:
            disc_text = "(need both lush and stunted patches to compute a\ndiscrimination index)"

        self.info_label.config(
            text=(f"t = {t:.2f}\n\n{disc_text}\n\n"
                  f"Compare to Kareiva's discrimination index = \n"
                  f"(mean beetles/lush patch) / (mean beetles/stunted patch),\n"
                  f"his Fig. 8: roughly 2-5 depending on spacing.")
        )


if __name__ == "__main__":
    app = HabitatSelectionApp()
    app.mainloop()