"""Cellular automaton experiments: rule outcomes over many seeds, and noise sweeps.

Rules are outer-totalistic with a Moore neighborhood on a torus.

Outcome classes (judged from the population history):
  extinct    - population is 0 at the end
  saturated  - mean population over the last 60 steps is above 85% of cells
  stable     - population constant over the last 60 steps
  periodic   - last 60 steps repeat with some period from 1 to 20
  active     - everything else
"""
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_v2")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2026)

LABELS = ["extinct", "stable", "periodic", "active", "saturated"]
COLORS = {"extinct": "#ff6b57", "stable": "#4fd6c4", "periodic": "#ffb000",
          "active": "#7a7a7a", "saturated": "#c98cff"}


# --- CA core ---

def make_tables(birth, survive):
    B = np.zeros(9, np.uint8)
    S = np.zeros(9, np.uint8)
    B[list(birth)] = 1
    S[list(survive)] = 1
    return B, S


def step(grid, B, S):
    counts = np.zeros(grid.shape, np.int8)
    for dy in (-1, 0, 1):
        shifted = np.roll(grid, dy, 0)
        for dx in (-1, 0, 1):
            if dx or dy:
                counts += np.roll(shifted, dx, 1)
    return np.where(grid == 1, S[counts], B[counts]).astype(np.uint8)


def run(grid, birth, survive, T, noise=0.0):
    B, S = make_tables(birth, survive)
    pops = np.empty(T, np.int32)
    for t in range(T):
        pops[t] = grid.sum()
        grid = step(grid, B, S)
        if noise > 0:
            flips = (rng.random(grid.shape) < noise).astype(np.uint8)
            grid = grid ^ flips
    return pops, grid


def classify(pops, cells, tail=60, max_period=20):
    if pops[-1] == 0:
        return "extinct"
    t = pops[-tail:]
    if t.mean() > 0.85 * cells:
        return "saturated"
    if t.max() == t.min():
        return "stable"
    for p in range(1, max_period + 1):
        if np.array_equal(t[p:], t[:-p]):
            return "periodic"
    return "active"


def random_rule():
    birth = [n for n in range(9) if rng.random() < 0.35]
    survive = [n for n in range(9) if rng.random() < 0.35]
    return birth, survive


def rule_name(b, s):
    return "B" + "".join(map(str, b)) + "/S" + "".join(map(str, s))


def majority(labels):
    return Counter(labels).most_common(1)[0][0]


# --- Experiment A: many random rules, 20 seeds each ---

def exp_rules(n_rules=100, n_seeds=20, N=48, T=300, density=0.35):
    results = []
    for _ in range(n_rules):
        b, s = random_rule()
        labels, dens = [], []
        for _ in range(n_seeds):
            g = (rng.random((N, N)) < density).astype(np.uint8)
            pops, _ = run(g, b, s, T)
            labels.append(classify(pops, N * N))
            dens.append(pops[-60:].mean() / (N * N))
        results.append({"rule": rule_name(b, s), "b": b, "s": s,
                        "labels": labels, "final_density": dens})

    with open(OUT + "/rules_20seeds.json", "w") as f:
        json.dump(results, f)

    tally20 = Counter(majority(r["labels"]) for r in results)
    tally3 = Counter(majority(r["labels"][:3]) for r in results)
    changed = sum(majority(r["labels"][:3]) != majority(r["labels"]) for r in results)
    purity = np.array([Counter(r["labels"]).most_common(1)[0][1] / n_seeds
                       for r in results])
    mixed = int((purity < 0.8).sum())  # dominant label under 80% of seeds

    print("20-seed majority tally:", dict(tally20))
    print("3-seed majority tally :", dict(tally3))
    print(f"label changes between 3 and 20 seeds: {changed}/{n_rules}")
    print(f"initial-condition-sensitive rules: {mixed}/{n_rules}")

    # A1: 3 seeds vs 20 seeds
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.38
    x = np.arange(len(LABELS))
    ax.bar(x - w / 2, [tally3[l] for l in LABELS], w, label="3 seeds/rule", color="#bbbbbb")
    ax.bar(x + w / 2, [tally20[l] for l in LABELS], w, label="20 seeds/rule",
           color=[COLORS[l] for l in LABELS])
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel(f"# rules (of {n_rules})")
    ax.set_title(f"Majority label, 3 vs 20 seeds\n"
                 f"{changed} rules change label; {mixed} are IC-sensitive")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT + "/A1_tally_3_vs_20.png", dpi=150)
    plt.close(fig)

    # A2: label fractions per rule
    order = sorted(range(n_rules),
                   key=lambda i: (LABELS.index(majority(results[i]["labels"])), -purity[i]))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    bottom = np.zeros(n_rules)
    for lab in LABELS:
        frac = np.array([Counter(results[i]["labels"])[lab] / n_seeds for i in order])
        ax.bar(range(n_rules), frac, bottom=bottom, color=COLORS[lab], label=lab, width=1.0)
        bottom += frac
    ax.set_xlabel("rule (sorted by dominant outcome)")
    ax.set_ylabel(f"fraction of {n_seeds} seeds")
    ax.set_title(f"Outcome distribution per rule over {n_seeds} random starts")
    ax.legend(ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.5))
    fig.tight_layout()
    fig.savefig(OUT + "/A2_per_rule_distribution.png", dpi=150)
    plt.close(fig)

    # A3: final density per rule
    mean_d = np.array([np.mean(results[i]["final_density"]) for i in order])
    sd_d = np.array([np.std(results[i]["final_density"]) for i in order])
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.errorbar(range(n_rules), mean_d * 100, yerr=sd_d * 100,
                fmt="o", ms=3, lw=0.8, color="#333")
    ax.set_xlabel("rule (same order as A2)")
    ax.set_ylabel("final density (%)")
    ax.set_title(f"Mean final density per rule (bars: sd over {n_seeds} seeds)")
    fig.tight_layout()
    fig.savefig(OUT + "/A3_final_density_per_rule.png", dpi=150)
    plt.close(fig)
    return results


# --- Experiment B: noise sweep on a few named rules ---

RULES = {
    "Life B3/S23": ([3], [2, 3]),
    "HighLife B36/S23": ([3, 6], [2, 3]),
    "Day&Night B3678/S34678": ([3, 6, 7, 8], [3, 4, 6, 7, 8]),
    "Maze B3/S12345": ([3], [1, 2, 3, 4, 5]),
}


def exp_noise_field(N=64, T=300, trials=16, density=0.35):
    rates = np.concatenate([[0.0], np.logspace(-4, np.log10(0.2), 14)])
    data = {}
    for name, (b, s) in RULES.items():
        rows = []
        for nr in rates:
            vals = []
            for _ in range(trials):
                g = (rng.random((N, N)) < density).astype(np.uint8)
                pops, _ = run(g, b, s, T, noise=nr)
                vals.append(pops[-50:].mean() / (N * N))
            rows.append(vals)
        data[name] = np.array(rows)
        print(name, "median density % by noise:",
              [f"{100 * np.median(r):.1f}" for r in rows])

    dump = {k: v.tolist() for k, v in data.items()}
    dump["rates"] = rates.tolist()
    with open(OUT + "/noise_field.json", "w") as f:
        json.dump(dump, f)

    xs = np.where(rates == 0, 3e-5, rates)  # put zero noise at the left of the log axis
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, arr in data.items():
        med = np.median(arr, 1) * 100
        lo = np.percentile(arr, 25, 1) * 100
        hi = np.percentile(arr, 75, 1) * 100
        axes[0].plot(xs, med, marker="o", ms=4, label=name)
        axes[0].fill_between(xs, lo, hi, alpha=0.15)
        axes[1].plot(xs, arr.std(1) * 100, marker="o", ms=4, label=name)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("noise probability per cell per step (leftmost = 0)")
        ax.axvline(5e-5, color="#aaa", ls=":", lw=0.8)
    axes[0].set_ylabel(f"final density (%), median and IQR over {trials} runs")
    axes[0].set_title("Outcome vs. noise level")
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel("sd of final density across runs (%)")
    axes[1].set_title("Run-to-run variability")
    fig.tight_layout()
    fig.savefig(OUT + "/B1_density_vs_noise.png", dpi=150)
    plt.close(fig)

    # B2: Life outcomes binned by final density
    arr = data["Life B3/S23"]
    bins = [("dead (<0.1%)", 0, 0.001), ("sparse (0.1-5%)", 0.001, 0.05),
            ("moderate (5-20%)", 0.05, 0.20), ("dense (>20%)", 0.20, 1.01)]
    bin_colors = ["#ff6b57", "#4fd6c4", "#ffb000", "#c98cff"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bottom = np.zeros(len(rates))
    for (lab, lo, hi), c in zip(bins, bin_colors):
        frac = ((arr >= lo) & (arr < hi)).mean(1)
        ax.bar(range(len(rates)), frac, bottom=bottom, color=c, label=lab)
        bottom += frac
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(["0"] + [f"{r:.0e}" for r in rates[1:]], rotation=60, fontsize=7)
    ax.set_xlabel("noise probability")
    ax.set_ylabel(f"fraction of {trials} runs")
    ax.set_title("Life: final-density outcomes vs noise")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(OUT + "/B2_life_outcome_bins.png", dpi=150)
    plt.close(fig)


# --- Experiment C: small seeded patterns vs an empty-grid control ---

PATTERNS = {
    "glider": np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], np.uint8),
    "blinker": np.array([[1, 1, 1]], np.uint8),
    "block": np.array([[1, 1], [1, 1]], np.uint8),
}


def exp_noise_patterns(N=40, T=200, trials=30):
    rates = np.concatenate([[0.0], np.logspace(-4, -1.5, 9)])
    xs = np.where(rates == 0, 3e-5, rates)

    # control: empty grid, so any density comes from noise alone
    control = []
    for nr in rates:
        vals = []
        for _ in range(trials):
            pops, _ = run(np.zeros((N, N), np.uint8), [3], [2, 3], T, noise=nr)
            vals.append(pops[-30:].mean() / (N * N))
        control.append(vals)
    control = np.array(control)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(xs, control.mean(1) * 100, "k--", marker="x", label="empty grid (noise only)")

    out = {}
    for name, pat in PATTERNS.items():
        rows = []
        for nr in rates:
            vals = []
            for _ in range(trials):
                g = np.zeros((N, N), np.uint8)
                y = x = N // 2
                g[y:y + pat.shape[0], x:x + pat.shape[1]] = pat
                pops, _ = run(g, [3], [2, 3], T, noise=nr)
                vals.append(pops[-30:].mean() / (N * N))
            rows.append(vals)
        rows = np.array(rows)
        out[name] = rows
        ax.errorbar(xs, rows.mean(1) * 100, yerr=rows.std(1) / np.sqrt(trials) * 100,
                    marker="o", ms=4, capsize=2, label=name)

    ax.set_xscale("log")
    ax.set_xlabel("noise probability (leftmost = 0)")
    ax.set_ylabel(f"final density (%), mean +/- s.e.m. ({trials} runs)")
    ax.set_title("Seeded pattern vs. empty-grid control under noise\n"
                 "(a pattern is robust only where it departs from the dashed control)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT + "/C1_pattern_density_vs_noise.png", dpi=150)
    plt.close(fig)

    for name, rows in out.items():
        print(name, "mean density %:", [f"{100 * r.mean():.2f}" for r in rows])
    print("control mean density %:", [f"{100 * r.mean():.2f}" for r in control])


if __name__ == "__main__":
    exp_rules()
    exp_noise_field()
    exp_noise_patterns()
