"""
Experiments backing the written report for the Emergent Complexity assignment.
Runs headless (numpy) simulations of outer-totalistic 2D cellular automata,
identical rule semantics to the website (Moore neighborhood, toroidal wrap).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, random

rng = np.random.default_rng(7)

def neighbor_count(grid):
    # sum of the 8 Moore neighbors via toroidal shifts
    s = np.zeros_like(grid, dtype=np.int8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            s += np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
    return s

def step(grid, birth, survive):
    n = neighbor_count(grid)
    birth_mask = np.isin(n, list(birth))
    survive_mask = np.isin(n, list(survive))
    new = np.where(grid == 1, survive_mask, birth_mask).astype(np.uint8)
    return new

def run(grid, birth, survive, steps, noise_rate=0.0, record_every=1):
    pops = []
    frames = []
    for t in range(steps):
        pops.append(int(grid.sum()))
        if t % record_every == 0:
            frames.append(grid.copy())
        grid = step(grid, birth, survive)
        if noise_rate > 0:
            flips = rng.random(grid.shape) < noise_rate
            grid = np.where(flips, 1 - grid, grid).astype(np.uint8)
    return np.array(pops), frames

def classify(pops, total_cells):
    last = pops[-1]
    if last == 0:
        return "extinct"
    if last > 0.85 * total_cells:
        return "saturated"
    tail = pops[-30:] if len(pops) >= 30 else pops
    mean = tail.mean()
    cv = tail.std() / mean if mean > 0 else 0
    if cv < 0.01:
        return "stable"
    if cv < 0.15:
        return "oscillating"
    return "chaotic"

def random_rule():
    b = [n for n in range(9) if rng.random() < 0.35]
    s = [n for n in range(9) if rng.random() < 0.35]
    return b, s

# ---------------------------------------------------------------
# TASK 1: density sweep for Conway's Game of Life
# ---------------------------------------------------------------
def task1_density_sweep():
    N = 80
    steps = 300
    densities = [0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for d in densities:
        grid = (rng.random((N, N)) < d).astype(np.uint8)
        pops, _ = run(grid, {3}, {2, 3}, steps)
        ax.plot(pops / (N * N) * 100, label=f"init density {int(d*100)}%")
    ax.set_xlabel("generation")
    ax.set_ylabel("population density (%)")
    ax.set_title("Conway's Life (B3/S23): population vs. time across initial densities")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig("/home/claude/ca-lab/fig_task1_density_sweep.png", dpi=150)
    plt.close(fig)

    # save one representative long run's grid at a few checkpoints as images
    N = 100
    grid = (rng.random((N, N)) < 0.35).astype(np.uint8)
    checkpoints = [0, 10, 50, 150, 400]
    steps_needed = max(checkpoints) + 1
    fig, axes = plt.subplots(1, len(checkpoints), figsize=(14, 3))
    g = grid.copy()
    saved = {}
    for t in range(steps_needed):
        if t in checkpoints:
            saved[t] = g.copy()
        g = step(g, {3}, {2, 3})
    for ax, t in zip(axes, checkpoints):
        ax.imshow(saved[t], cmap="copper", interpolation="nearest")
        ax.set_title(f"t={t}")
        ax.axis("off")
    fig.suptitle("Game of Life settling from 35% random noise into stable/oscillating debris")
    fig.tight_layout()
    fig.savefig("/home/claude/ca-lab/fig_task1_snapshots.png", dpi=150)
    plt.close(fig)
    print("Task 1 figures saved.")

# ---------------------------------------------------------------
# TASK 3: sample 100 random rules, classify
# ---------------------------------------------------------------
def task3_rule_sampling():
    N = 48
    steps = 150
    n_rules = 100
    n_ic_per_rule = 3
    results = []
    for i in range(n_rules):
        b, s = random_rule()
        labels_for_rule = []
        for _ in range(n_ic_per_rule):
            grid = (rng.random((N, N)) < 0.35).astype(np.uint8)
            pops, _ = run(grid, b, s, steps)
            labels_for_rule.append(classify(pops, N * N))
        results.append({"b": b, "s": s, "labels": labels_for_rule})

    # tally: use the majority label across the 3 ICs as the rule's category,
    # but flag rules whose behavior *depends* on initial condition
    from collections import Counter
    tally = Counter()
    sensitive = []
    for r in results:
        c = Counter(r["labels"])
        top_label, top_count = c.most_common(1)[0]
        tally[top_label] += 1
        if len(c) > 1:
            sensitive.append(r)

    with open("/home/claude/ca-lab/task3_results.json", "w") as f:
        json.dump(results, f)

    fig, ax = plt.subplots(figsize=(6, 4))
    cats = list(tally.keys())
    vals = [tally[c] for c in cats]
    colors = {"extinct": "#ff6b57", "stable": "#4fd6c4", "oscillating": "#ffb000",
              "saturated": "#c98cff", "chaotic": "#888888"}
    ax.bar(cats, vals, color=[colors.get(c, "#666") for c in cats])
    ax.set_ylabel("number of rules (out of 100)")
    ax.set_title("Behavior of 100 random outer-totalistic rules\n(majority label across 3 random initial conditions)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig("/home/claude/ca-lab/fig_task3_tally.png", dpi=150)
    plt.close(fig)

    print("Task 3 tally:", dict(tally))
    print(f"{len(sensitive)} of {n_rules} rules were initial-condition-sensitive "
          f"(different outcome categories across the 3 seeds).")

    return results, tally, sensitive

# ---------------------------------------------------------------
# Pick one interesting rule (oscillating/chaotic boundary) for closer study
# ---------------------------------------------------------------
def closer_look(rule_b, rule_s, name="candidate"):
    N = 90
    steps = 400
    grid = (rng.random((N, N)) < 0.35).astype(np.uint8)
    pops, frames = run(grid, rule_b, rule_s, steps, record_every=1)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(pops / (N * N) * 100, color="#ffb000")
    ax.set_xlabel("generation")
    ax.set_ylabel("population density (%)")
    ax.set_title(f"B{''.join(map(str,rule_b))}/S{''.join(map(str,rule_s))} — population over time")
    fig.tight_layout()
    fig.savefig(f"/home/claude/ca-lab/fig_closerlook_{name}_trace.png", dpi=150)
    plt.close(fig)

    checkpoints = [0, 20, 100, 250, 399]
    fig, axes = plt.subplots(1, len(checkpoints), figsize=(14, 3))
    for ax_, t in zip(axes, checkpoints):
        ax_.imshow(frames[t], cmap="copper", interpolation="nearest")
        ax_.set_title(f"t={t}")
        ax_.axis("off")
    fig.suptitle(f"B{''.join(map(str,rule_b))}/S{''.join(map(str,rule_s))} snapshots")
    fig.tight_layout()
    fig.savefig(f"/home/claude/ca-lab/fig_closerlook_{name}_snapshots.png", dpi=150)
    plt.close(fig)
    print(f"Closer-look figures for {name} saved. Final density: {pops[-1]/(N*N)*100:.1f}%")

# ---------------------------------------------------------------
# OPTION B: noise robustness of a few known Life patterns / rules
# ---------------------------------------------------------------
GLIDER = np.array([[0,1,0],[0,0,1],[1,1,1]], dtype=np.uint8)
BLINKER = np.array([[1,1,1]], dtype=np.uint8)
BLOCK = np.array([[1,1],[1,1]], dtype=np.uint8)

def place(grid, pattern, y, x):
    grid[y:y+pattern.shape[0], x:x+pattern.shape[1]] = pattern
    return grid

def noise_robustness_patterns():
    N = 40
    steps = 200
    noise_rates = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
    patterns = {"glider": GLIDER, "blinker": BLINKER, "block": BLOCK}
    survival = {name: [] for name in patterns}
    trials = 20
    for name, pat in patterns.items():
        for nr in noise_rates:
            alive_count = 0
            for _ in range(trials):
                grid = np.zeros((N, N), dtype=np.uint8)
                grid = place(grid, pat, N//2, N//2)
                pops, _ = run(grid, {3}, {2, 3}, steps, noise_rate=nr)
                # "survives" if population doesn't hit 0 by the end
                if pops[-1] > 0:
                    alive_count += 1
            survival[name].append(alive_count / trials * 100)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, vals in survival.items():
        ax.plot([r*100 for r in noise_rates], vals, marker="o", label=name)
    ax.set_xlabel("noise rate (% chance per cell per step)")
    ax.set_ylabel(f"% of {trials} trials with cells alive after {steps} steps")
    ax.set_title("Robustness of small Game-of-Life patterns to random bit-flip noise")
    ax.legend()
    fig.tight_layout()
    fig.savefig("/home/claude/ca-lab/fig_optionB_pattern_robustness.png", dpi=150)
    plt.close(fig)
    print("Option B pattern-robustness figure saved.")
    print(json.dumps(survival, indent=2))
    return survival

def noise_robustness_field():
    # whole-field random-start Life density, under increasing noise, does the
    # system find a "quiet" fixed point, or does noise keep it churning?
    N = 70
    steps = 400
    noise_rates = [0.0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    finalvals = []
    for nr in noise_rates:
        grid = (rng.random((N, N)) < 0.35).astype(np.uint8)
        pops, _ = run(grid, {3}, {2, 3}, steps, noise_rate=nr)
        ax.plot(pops / (N * N) * 100, label=f"noise {nr*100:g}%")
        finalvals.append(pops[-30:].mean() / (N * N) * 100)
    ax.set_xlabel("generation")
    ax.set_ylabel("population density (%)")
    ax.set_title("Game of Life under sustained random noise")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("/home/claude/ca-lab/fig_optionB_field_noise.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot([r*100 for r in noise_rates], finalvals, marker="o", color="#ff6b57")
    ax2.set_xlabel("noise rate (% per cell per step)")
    ax2.set_ylabel("steady-state density (%, last 30 gens)")
    ax2.set_title("Steady-state density vs. noise rate")
    fig2.tight_layout()
    fig2.savefig("/home/claude/ca-lab/fig_optionB_field_noise_threshold.png", dpi=150)
    plt.close(fig2)
    print("Option B field-noise figures saved.")
    print("final densities by noise rate:", dict(zip(noise_rates, finalvals)))

if __name__ == "__main__":
    task1_density_sweep()
    results, tally, sensitive = task3_rule_sampling()

    # pick an interesting rule: one whose 3 seeds disagreed (IC-sensitive),
    # preferring one classified as oscillating/chaotic at least once
    interesting = None
    for r in sensitive:
        if "oscillating" in r["labels"] or "chaotic" in r["labels"]:
            interesting = r
            break
    if interesting is None and sensitive:
        interesting = sensitive[0]
    if interesting is None:
        interesting = results[0]
    print("Chosen rule for closer look:", interesting["b"], interesting["s"], interesting["labels"])
    closer_look(interesting["b"], interesting["s"], name="chosen")

    # also always show HighLife as a well-known "interesting" reference rule
    closer_look([3, 6], [2, 3], name="highlife")

    noise_robustness_patterns()
    noise_robustness_field()
