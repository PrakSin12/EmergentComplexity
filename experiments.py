"""
experiments_v2.py -- revisions responding to George's feedback.
  1. Explicit, documented classifier definitions (CLASSIFIER_DOC below).
  2. 20 seeds per rule instead of 3, plus a comparison of 3-seed vs 20-seed conclusions.
  3. Graded measures (mean density over a final window) instead of "population > 0".
  4. Figure of outcome vs. noise probability across several rules.
Same rule semantics as the website: outer-totalistic, Moore neighborhood, toroidal wrap.
"""
import numpy as np, json, os
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__)) + "/fig_v2"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2026)

CLASSIFIER_DOC = """
All labels use the population series pop[t] (live-cell count) of one run of T=300 steps
on a 48x48 torus, started from i.i.d. 35% random noise. TAIL = last 60 generations.
  extinct    : pop[T-1] == 0.
  saturated  : mean tail density > 85%.
  stable     : pop is exactly constant over the tail (still lifes / everything frozen).
  periodic   : not stable, but pop is exactly periodic over the tail with some period p <= 20
               (e.g. blinkers/pulsars everywhere, or a rule with a global oscillation).
  active     : none of the above -- population keeps fluctuating non-periodically
               (gliders still travelling, growth fronts, disordered / chaotic churn).
Caveat: this is a population-only proxy; two different grids can share a population series.
"""

# ---------- core CA ----------
def make_tables(birth, survive):
    B = np.zeros(9, np.uint8); S = np.zeros(9, np.uint8)
    B[list(birth)] = 1; S[list(survive)] = 1
    return B, S

def step(g, B, S):
    n = np.zeros(g.shape, np.int8)
    for dy in (-1, 0, 1):
        r = np.roll(g, dy, 0)
        for dx in (-1, 0, 1):
            if dx or dy:
                n += np.roll(r, dx, 1)
    return np.where(g == 1, S[n], B[n]).astype(np.uint8)

def run(g, birth, survive, T, noise=0.0):
    B, S = make_tables(birth, survive)
    pops = np.empty(T, np.int32)
    for t in range(T):
        pops[t] = g.sum()
        g = step(g, B, S)
        if noise > 0:
            g = g ^ (rng.random(g.shape) < noise).astype(np.uint8)
    return pops, g

# ---------- classifier ----------
def classify(pops, cells, tail=60, maxp=20):
    if pops[-1] == 0: return "extinct"
    t = pops[-tail:]
    if t.mean() > 0.85 * cells: return "saturated"
    if t.max() == t.min(): return "stable"
    for p in range(1, maxp + 1):
        if np.array_equal(t[p:], t[:-p]): return "periodic"
    return "active"

LABELS = ["extinct", "stable", "periodic", "active", "saturated"]
COLORS = {"extinct": "#ff6b57", "stable": "#4fd6c4", "periodic": "#ffb000",
          "active": "#7a7a7a", "saturated": "#c98cff"}

def random_rule():
    return ([n for n in range(9) if rng.random() < 0.35],
            [n for n in range(9) if rng.random() < 0.35])

def fmt(b, s): return "B" + "".join(map(str, b)) + "/S" + "".join(map(str, s))

# ---------- Experiment A: 100 rules x 20 seeds ----------
def exp_rules(n_rules=100, n_seeds=20, N=48, T=300, density=0.35):
    res = []
    for _ in range(n_rules):
        b, s = random_rule()
        labs, dens = [], []
        for _ in range(n_seeds):
            g = (rng.random((N, N)) < density).astype(np.uint8)
            pops, _ = run(g, b, s, T)
            labs.append(classify(pops, N * N))
            dens.append(pops[-60:].mean() / (N * N))
        res.append(dict(rule=fmt(b, s), b=b, s=s, labels=labs, final_density=dens))
    json.dump(res, open(OUT + "/rules_20seeds.json", "w"))

    def majority(l):
        return Counter(l).most_common(1)[0][0]
    t20 = Counter(majority(r["labels"]) for r in res)
    t3 = Counter(majority(r["labels"][:3]) for r in res)
    disagree = sum(majority(r["labels"][:3]) != majority(r["labels"]) for r in res)
    purity = np.array([Counter(r["labels"]).most_common(1)[0][1] / n_seeds for r in res])
    mixed = int((purity < 0.8).sum())  # IC-sensitive: dominant label < 80% of seeds
    print("20-seed majority tally:", dict(t20))
    print("3-seed majority tally :", dict(t3))
    print(f"rules whose label changes when using 3 vs 20 seeds: {disagree}/{n_rules}")
    print(f"IC-sensitive rules (dominant label <80% of {n_seeds} seeds): {mixed}/{n_rules}")

    # Fig A1: 3 seeds vs 20 seeds tally
    fig, ax = plt.subplots(figsize=(7, 4)); w = 0.38; x = np.arange(len(LABELS))
    ax.bar(x - w/2, [t3[l] for l in LABELS], w, label="3 seeds/rule", color="#bbbbbb")
    ax.bar(x + w/2, [t20[l] for l in LABELS], w, label="20 seeds/rule",
           color=[COLORS[l] for l in LABELS])
    ax.set_xticks(x); ax.set_xticklabels(LABELS); ax.set_ylabel("# rules (of 100)")
    ax.set_title(f"Majority-label tally: 3 vs 20 seeds\n{disagree} rules change label; {mixed} are IC-sensitive")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT + "/A1_tally_3_vs_20.png", dpi=150); plt.close(fig)

    # Fig A2: per-rule stacked label fractions, sorted by dominant label
    order = sorted(range(n_rules), key=lambda i: (LABELS.index(majority(res[i]["labels"])), -purity[i]))
    fig, ax = plt.subplots(figsize=(11, 3.6)); bottom = np.zeros(n_rules)
    for lab in LABELS:
        f = np.array([Counter(res[i]["labels"])[lab] / n_seeds for i in order])
        ax.bar(range(n_rules), f, bottom=bottom, color=COLORS[lab], label=lab, width=1.0); bottom += f
    ax.set_xlabel("rule (sorted by dominant outcome)"); ax.set_ylabel("fraction of 20 seeds")
    ax.set_title("Per-rule outcome distribution over 20 random initial conditions")
    ax.legend(ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.5))
    fig.tight_layout(); fig.savefig(OUT + "/A2_per_rule_distribution.png", dpi=150); plt.close(fig)

    # Fig A3: graded view -- final density per rule (mean +/- sd across seeds)
    md = np.array([np.mean(res[i]["final_density"]) for i in order])
    sd = np.array([np.std(res[i]["final_density"]) for i in order])
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.errorbar(range(n_rules), md * 100, yerr=sd * 100, fmt="o", ms=3, lw=0.8, color="#333")
    ax.set_xlabel("rule (same order as above)"); ax.set_ylabel("final density (%)")
    ax.set_title("Graded measure: mean final density per rule (error bar = sd across 20 seeds)")
    fig.tight_layout(); fig.savefig(OUT + "/A3_final_density_per_rule.png", dpi=150); plt.close(fig)
    return res

# ---------- Experiment B: noise sweep (graded) ----------
RULES = {"Life B3/S23": ([3], [2, 3]), "HighLife B36/S23": ([3, 6], [2, 3]),
         "Day&Night B3678/S34678": ([3, 6, 7, 8], [3, 4, 6, 7, 8]),
         "Maze B3/S12345": ([3], [1, 2, 3, 4, 5])}

def exp_noise_field(N=64, T=300, trials=16, density=0.35):
    rates = np.logspace(-4, np.log10(0.2), 14)
    rates = np.concatenate([[0.0], rates])
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
              [f"{100*np.median(r):.1f}" for r in rows])
    json.dump({k: v.tolist() for k, v in data.items()} | {"rates": rates.tolist()},
              open(OUT + "/noise_field.json", "w"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    xs = np.where(rates == 0, 3e-5, rates)  # plot noise=0 at left edge of log axis
    for name, arr in data.items():
        med = np.median(arr, 1) * 100
        lo, hi = np.percentile(arr, 25, 1) * 100, np.percentile(arr, 75, 1) * 100
        axes[0].plot(xs, med, marker="o", ms=4, label=name)
        axes[0].fill_between(xs, lo, hi, alpha=0.15)
        axes[1].plot(xs, arr.std(1) * 100, marker="o", ms=4, label=name)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("noise probability per cell per step (leftmost = 0)")
        ax.axvline(5e-5, color="#aaa", ls=":", lw=0.8)
    axes[0].set_ylabel("final density (%), median & IQR over 16 runs")
    axes[0].set_title("Outcome vs. noise level"); axes[0].legend(fontsize=7)
    axes[1].set_ylabel("sd of final density across runs (%)")
    axes[1].set_title("Run-to-run variability (peaks suggest a transition)")
    fig.tight_layout(); fig.savefig(OUT + "/B1_density_vs_noise.png", dpi=150); plt.close(fig)

    # Fig B2: classifier-style breakdown vs noise for Life, using graded bins on density
    arr = data["Life B3/S23"]
    bins = [("dead (<0.1%)", 0, 0.001), ("sparse (0.1-5%)", 0.001, 0.05),
            ("moderate (5-20%)", 0.05, 0.20), ("dense (>20%)", 0.20, 1.01)]
    fig, ax = plt.subplots(figsize=(7.5, 4)); bottom = np.zeros(len(rates))
    for (lab, lo, hi), c in zip(bins, ["#ff6b57", "#4fd6c4", "#ffb000", "#c98cff"]):
        f = ((arr >= lo) & (arr < hi)).mean(1)
        ax.bar(range(len(rates)), f, bottom=bottom, color=c, label=lab); bottom += f
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(["0"] + [f"{r:.0e}" for r in rates[1:]], rotation=60, fontsize=7)
    ax.set_xlabel("noise probability"); ax.set_ylabel("fraction of 16 runs")
    ax.set_title("Life: distribution of final-density outcomes vs noise")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1)); fig.tight_layout()
    fig.savefig(OUT + "/B2_life_outcome_bins.png", dpi=150); plt.close(fig)

# ---------- Experiment C: small patterns, graded, with noise-only control ----------
PATTERNS = {"glider": np.array([[0,1,0],[0,0,1],[1,1,1]], np.uint8),
            "blinker": np.array([[1,1,1]], np.uint8),
            "block": np.array([[1,1],[1,1]], np.uint8)}

def exp_noise_patterns(N=40, T=200, trials=30):
    rates = np.concatenate([[0.0], np.logspace(-4, -1.5, 9)])
    fig, ax = plt.subplots(figsize=(7.5, 4.2)); out = {}
    ctrl = []
    for nr in rates:  # control: empty grid + same noise -> baseline density from noise alone
        v = []
        for _ in range(trials):
            pops, _ = run(np.zeros((N, N), np.uint8), [3], [2, 3], T, noise=nr)
            v.append(pops[-30:].mean() / (N * N))
        ctrl.append(v)
    ctrl = np.array(ctrl); xs = np.where(rates == 0, 3e-5, rates)
    ax.plot(xs, ctrl.mean(1) * 100, "k--", marker="x", label="empty grid (noise only)")
    for name, pat in PATTERNS.items():
        rows = []
        for nr in rates:
            v = []
            for _ in range(trials):
                g = np.zeros((N, N), np.uint8); y = x = N // 2
                g[y:y+pat.shape[0], x:x+pat.shape[1]] = pat
                pops, _ = run(g, [3], [2, 3], T, noise=nr)
                v.append(pops[-30:].mean() / (N * N))
            rows.append(v)
        rows = np.array(rows); out[name] = rows
        ax.errorbar(xs, rows.mean(1) * 100, yerr=rows.std(1) / np.sqrt(trials) * 100,
                    marker="o", ms=4, capsize=2, label=name)
    ax.set_xscale("log"); ax.set_xlabel("noise probability (leftmost = 0)")
    ax.set_ylabel("final density (%), mean +/- s.e.m. (30 runs)")
    ax.set_title("Seeded pattern vs. empty-grid control under noise\n(pattern is 'robust' only where its curve departs from the dashed control)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(OUT + "/C1_pattern_density_vs_noise.png", dpi=150); plt.close(fig)
    for name, rows in out.items():
        print(name, "mean density % :", [f"{100*r.mean():.2f}" for r in rows])
    print("control mean density % :", [f"{100*r.mean():.2f}" for r in ctrl])

if __name__ == "__main__":
    print(CLASSIFIER_DOC)
    exp_rules()
    exp_noise_field()
    exp_noise_patterns()
