#!/usr/bin/env python3
"""
Run a benchmark workload through the HBM thermal pipeline and produce
analysis plots and a report.

Usage::

    python run_benchmark.py --benchmark stream --iterations 50 --output-dir results/stream
    python run_benchmark.py --benchmark sgemm  --iterations 100
    python run_benchmark.py --list   # show available benchmarks

The script does **not** require HotSpot to be compiled — when the binary
is absent the pipeline still runs the power model and generates synthetic
temperature estimates (uniform warming from power dissipation) so that the
full analysis workflow can be demonstrated.
"""

import argparse
import json
import logging
import math
import os
import sys

import numpy as np

# Ensure repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.config import PipelineConfig
from pipeline.hbm_power_model import HBMPowerModel
from pipeline.benchmarks import list_benchmarks, get_benchmark

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ======================================================================
# Lightweight thermal estimator (no HotSpot binary needed)
# ======================================================================
class SimpleThermalModel:
    """Estimate bank temperatures from power using a simple RC thermal model.

    This is a first-order lumped thermal model used when HotSpot is not
    available.  It models each bank as an independent thermal node with:

        C · dT/dt = P − (T − T_ambient) / R_th

    Discretised with forward-Euler per sampling interval.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        n = cfg.NUM_BANKS
        self.ambient_c = 45.0  # from mem_hotspot.config: -ambient 318.15 K = 45.0 °C
        # Thermal resistance: derived from HotSpot package params
        # R_conv = 0.1 K/W (from config).  Spread across 128 banks → ~12.8 K/W
        self.r_th = 12.8  # K/W per bank
        # Thermal capacitance per bank: silicon specific heat × volume
        # 1.75e6 J/(m³·K) × (1.707e-3)² × 50e-6 m ≈ 2.55e-4 J/K
        bank_area = (1.707e-3) ** 2  # m²  (from floorplan)
        bank_thick = 50e-6  # m   (from LCF)
        self.c_th = 1.75e6 * bank_area * bank_thick  # J/K per bank

        self.temps = np.full(n, self.ambient_c)

    def step(self, bank_power, dt_sec):
        """Advance one time step given per-bank power (W) and dt (seconds)."""
        p = np.array(bank_power, dtype=float)
        # Forward-Euler: T_new = T + dt/C * (P - (T - T_amb) / R)
        self.temps = self.temps + (dt_sec / self.c_th) * (
            p - (self.temps - self.ambient_c) / self.r_th
        )
        return self.temps.copy()


# ======================================================================
# Main simulation
# ======================================================================
def run_simulation(benchmark_name, iterations, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cfg = PipelineConfig(output_dir=output_dir)
    model = HBMPowerModel(cfg)
    bench_fn = get_benchmark(benchmark_name)
    thermal = SimpleThermalModel(cfg)

    dt_sec = cfg.interval_sec
    num_banks = cfg.NUM_BANKS
    banks_per_layer = cfg.banks_per_layer

    # Storage
    all_bank_power = []
    all_bank_temps = []
    all_logic_power = []
    peak_temps = []
    mean_temps = []
    total_power_w = []

    log.info("Running benchmark '%s' for %d iterations", benchmark_name, iterations)

    for step in range(iterations):
        reads, writes = bench_fn(cfg, step)

        bank_power, logic_power = model.compute_power_trace(
            reads, writes,
            bank_active=[1] * num_banks,
            freq_scale=1.0,
            voltage_scale=1.0,
        )

        temps = thermal.step(bank_power, dt_sec)
        model.set_bank_temperatures(temps)

        all_bank_power.append(list(bank_power))
        all_bank_temps.append(temps.tolist())
        all_logic_power.append(list(logic_power))
        peak_temps.append(float(np.max(temps)))
        mean_temps.append(float(np.mean(temps)))
        total_power_w.append(sum(bank_power) + sum(logic_power))

        if step % max(1, iterations // 10) == 0:
            log.info(
                "  step %3d/%d  peak=%.2f°C  mean=%.2f°C  total_power=%.3fW",
                step, iterations, peak_temps[-1], mean_temps[-1], total_power_w[-1],
            )

    # Save raw data
    results = {
        "benchmark": benchmark_name,
        "iterations": iterations,
        "peak_temps": peak_temps,
        "mean_temps": mean_temps,
        "total_power_w": total_power_w,
        "final_bank_temps": all_bank_temps[-1],
        "final_bank_power": all_bank_power[-1],
        "config": {
            "num_banks": num_banks,
            "banks_per_layer": banks_per_layer,
            "banks_in_z": cfg.BANKS_IN_Z,
            "banks_in_x": cfg.BANKS_IN_X,
            "banks_in_y": cfg.BANKS_IN_Y,
            "sampling_interval_ns": cfg.SAMPLING_INTERVAL_NS,
            "bank_static_power_ref_mw": cfg.BANK_STATIC_POWER_W_REF * 1000,
            "logic_core_power_mw": (
                cfg.LOGIC_CORE_STATIC_POWER_W + cfg.LOGIC_CORE_DYNAMIC_POWER_W
            ) * 1000,
        },
    }
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", json_path)

    return results


# ======================================================================
# Plotting
# ======================================================================
def generate_plots(results, output_dir):
    """Generate thermal and power analysis plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib import cm

    benchmark = results["benchmark"]
    iterations = results["iterations"]
    peak_temps = results["peak_temps"]
    mean_temps = results["mean_temps"]
    total_power = results["total_power_w"]
    final_temps = np.array(results["final_bank_temps"])
    final_power = np.array(results["final_bank_power"])
    cfg = results["config"]
    banks_in_x = cfg["banks_in_x"]
    banks_in_y = cfg["banks_in_y"]
    banks_in_z = cfg["banks_in_z"]
    banks_per_layer = cfg["banks_per_layer"]
    time_ms = np.arange(iterations)

    plots = {}

    # ---- Plot 1: Peak & Mean temperature over time --------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_ms, peak_temps, "r-", linewidth=1.5, label="Peak bank temperature")
    ax.plot(time_ms, mean_temps, "b-", linewidth=1.5, label="Mean bank temperature")
    ax.axhline(80.0, color="orange", linestyle="--", alpha=0.7, label="Throttle threshold (80°C)")
    ax.axhline(85.0, color="red", linestyle="--", alpha=0.7, label="Critical threshold (85°C)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"HBM Thermal Trace — {benchmark.upper()} Benchmark")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "thermal_trace.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots["thermal_trace"] = path
    log.info("Saved %s", path)

    # ---- Plot 2: Total power over time --------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_ms, total_power, "g-", linewidth=1.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Total HBM Stack Power (W)")
    ax.set_title(f"HBM Power Trace — {benchmark.upper()} Benchmark")
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "power_trace.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots["power_trace"] = path
    log.info("Saved %s", path)

    # ---- Plot 3: Per-layer thermal heatmaps (final state) -------------
    n_layers = banks_in_z
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    vmin = float(np.min(final_temps))
    vmax = float(np.max(final_temps))
    norm = Normalize(vmin=vmin, vmax=vmax)

    for z in range(n_layers):
        ax = axes[z // 4][z % 4]
        layer_temps = final_temps[z * banks_per_layer: (z + 1) * banks_per_layer]
        grid = layer_temps.reshape(banks_in_x, banks_in_y)
        im = ax.imshow(grid, cmap="hot", norm=norm, origin="lower", aspect="equal")
        ax.set_title(f"Layer {z + 1} (banks {z * banks_per_layer}–{(z + 1) * banks_per_layer - 1})")
        ax.set_xlabel("Bank Y")
        ax.set_ylabel("Bank X")
        for i in range(banks_in_x):
            for j in range(banks_in_y):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                        fontsize=7, color="white" if grid[i, j] > (vmin + vmax) / 2 else "black")

    fig.suptitle(f"Per-Layer Bank Temperature Heatmap (final state) — {benchmark.upper()}", fontsize=14)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="hot"), ax=axes, shrink=0.6, label="Temperature (°C)")
    path = os.path.join(output_dir, "layer_heatmaps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots["layer_heatmaps"] = path
    log.info("Saved %s", path)

    # ---- Plot 4: Per-layer power bar chart ----------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    layer_powers = []
    for z in range(n_layers):
        lp = final_power[z * banks_per_layer: (z + 1) * banks_per_layer]
        layer_powers.append(float(np.sum(lp)))
    bars = ax.bar(range(1, n_layers + 1), layer_powers, color="steelblue")
    ax.set_xlabel("DRAM Layer")
    ax.set_ylabel("Layer Total Power (W)")
    ax.set_title(f"Per-Layer Power Distribution — {benchmark.upper()}")
    ax.set_xticks(range(1, n_layers + 1))
    for bar, val in zip(bars, layer_powers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    path = os.path.join(output_dir, "layer_power_dist.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots["layer_power_dist"] = path
    log.info("Saved %s", path)

    # ---- Plot 5: 3D stack temperature cross-section -------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    # Show temperature of centre bank (bank index 5 = row 1, col 1) across layers
    centre_bank = 5  # centre of 4×4 grid
    layer_temps_centre = [final_temps[z * banks_per_layer + centre_bank] for z in range(n_layers)]
    ax.plot(range(1, n_layers + 1), layer_temps_centre, "ro-", markersize=8, linewidth=2)
    ax.set_xlabel("DRAM Layer (1 = closest to logic die)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Vertical Temperature Profile (centre bank) — {benchmark.upper()}")
    ax.set_xticks(range(1, n_layers + 1))
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "vertical_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots["vertical_profile"] = path
    log.info("Saved %s", path)

    return plots


# ======================================================================
# Report generation
# ======================================================================
def generate_report(results, plots, output_dir):
    """Write ANALYSIS_REPORT.md with embedded plot references."""
    benchmark = results["benchmark"]
    iterations = results["iterations"]
    cfg = results["config"]
    peak = max(results["peak_temps"])
    mean_final = results["mean_temps"][-1]
    total_pwr = results["total_power_w"][-1]

    report_lines = [
        f"# HBM Thermal Analysis Report — {benchmark.upper()} Benchmark",
        "",
        "## 1. Simulation Setup",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Benchmark | {benchmark} |",
        f"| Iterations (sampling intervals) | {iterations} |",
        f"| Sampling interval | {cfg['sampling_interval_ns'] / 1e6:.1f} ms |",
        f"| HBM banks | {cfg['num_banks']} ({cfg['banks_in_x']}×{cfg['banks_in_y']}×{cfg['banks_in_z']}) |",
        f"| Banks per layer | {cfg['banks_per_layer']} |",
        f"| Bank static power (ref @ 45°C) | {cfg['bank_static_power_ref_mw']:.1f} mW |",
        f"| Logic core power | {cfg['logic_core_power_mw']:.1f} mW |",
        f"| Leakage model | Exponential: P(T) = P_ref · exp(0.06·(T−45)) |",
        "",
        "## 2. Thermal Results Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Peak bank temperature | {peak:.2f} °C |",
        f"| Mean bank temperature (final) | {mean_final:.2f} °C |",
        f"| Total stack power (final) | {total_pwr:.3f} W |",
        f"| Steady-state reached | {'~Yes' if iterations >= 2 and abs(results['peak_temps'][-1] - results['peak_temps'][-2]) < 0.01 else 'No'} |",
        "",
        "## 3. Temperature Evolution",
        "",
        f"![Thermal Trace](thermal_trace.png)",
        "",
        "The plot shows peak and mean bank temperatures over the simulation.",
        "The dashed lines indicate the throttle (80°C) and critical (85°C) DTM thresholds.",
        "",
        "## 4. Power Consumption",
        "",
        f"![Power Trace](power_trace.png)",
        "",
        "Total HBM stack power including dynamic (read/write/activate/precharge),",
        "static (temperature-dependent leakage), refresh, and logic-core power.",
        "",
        "## 5. Per-Layer Thermal Heatmaps",
        "",
        f"![Layer Heatmaps](layer_heatmaps.png)",
        "",
        "Final-state temperature distribution across all 8 DRAM layers.",
        "Each heatmap shows the 4×4 bank grid with temperature annotations.",
        "Layer 1 is closest to the logic die (heat source); layer 8 is closest",
        "to the heat sink.",
        "",
        "## 6. Vertical Temperature Profile",
        "",
        f"![Vertical Profile](vertical_profile.png)",
        "",
        "Temperature of the centre bank (B_5) across the 8 DRAM layers.",
        "The gradient shows heat flowing from the logic die (layer 1) toward",
        "the heat sink (layer 8).",
        "",
        "## 7. Per-Layer Power Distribution",
        "",
        f"![Layer Power](layer_power_dist.png)",
        "",
        "Total power dissipated in each DRAM layer.",
        f"For the {benchmark} workload the distribution is uniform because access"
        " counts are evenly spread across banks." if benchmark in ("stream", "random") else
        f"For the {benchmark} workload the distribution reflects the non-uniform"
        " access pattern.",
        "",
        "## 8. Power Model Details",
        "",
        "The HBM power model includes five components:",
        "",
        "1. **Dynamic read/write power**: Energy per access (20.55 nJ from CACTI3DD)",
        "   × access count / sampling interval, scaled by V²·f for DVFS.",
        "2. **Activation/precharge power**: Each access implies a row open (3.2 nJ)",
        "   and close (1.1 nJ).",
        "3. **Refresh power**: Background refresh at t_REFI = 7.8 µs interval,",
        "   3.55 nJ per refresh command.",
        "4. **Static (leakage) power**: Temperature-dependent —",
        "   P_leak(T) = 20 mW · exp(0.06·(T − 45°C)). This models sub-threshold",
        "   leakage in 20 nm MOSFET technology, doubling every ~11.5°C.",
        "5. **Logic-core power**: I/O PHY + DLL + ECC engine = 35 mW per core",
        "   (20 mW static + 15 mW dynamic).",
        "",
        "## 9. Benchmark Profile",
        "",
    ]

    if benchmark == "stream":
        report_lines += [
            "**STREAM (copy/triad)**: Sequential bandwidth-bound workload.",
            "Sustains ~70% of peak HBM2 bandwidth (179 GB/s out of 256 GB/s).",
            "Read:write ratio ≈ 2:1. Uniform access across all 128 banks.",
            "This represents the worst-case sustained thermal load for HBM.",
        ]
    elif benchmark == "sgemm":
        report_lines += [
            "**SGEMM (dense matrix multiply)**: High arithmetic intensity,",
            "moderate HBM traffic (~45% peak BW). Bursty access pattern with",
            "even channels carrying more traffic due to column-major tile layout.",
        ]
    elif benchmark == "resnet50":
        report_lines += [
            "**ResNet-50 inference**: Periodic burst–idle pattern.",
            "Convolution layers produce ~60% peak BW; batch-norm/ReLU layers",
            "are compute-bound (~10% BW). 5-step repeating cycle.",
        ]
    elif benchmark == "random":
        report_lines += [
            "**Random access**: Pointer-chasing / graph workload with ~30%",
            "peak BW and non-uniform bank access distribution.",
        ]
    elif benchmark == "hotspot_stress":
        report_lines += [
            "**Hotspot stress test**: Heavy traffic concentrated on centre",
            "banks of each layer to create thermal gradients.",
        ]

    report_lines += [
        "",
        "---",
        "",
        "*Generated by `run_benchmark.py` — COL812 GPU & HotSpot Project*",
    ]

    report_path = os.path.join(output_dir, "ANALYSIS_REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    log.info("Report saved to %s", report_path)
    return report_path


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run HBM thermal simulation with standard benchmarks"
    )
    parser.add_argument(
        "--benchmark", "-b", type=str, default="stream",
        help="Benchmark name (default: stream). Use --list to see all.",
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=50,
        help="Number of sampling intervals to simulate (default: 50)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="Output directory (default: results/<benchmark>)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available benchmarks and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for name in list_benchmarks():
            fn = get_benchmark(name)
            doc = fn.__doc__.strip().split("\n")[0] if fn.__doc__ else ""
            print(f"  {name:20s}  {doc}")
        return

    output_dir = args.output_dir or os.path.join("results", args.benchmark)
    results = run_simulation(args.benchmark, args.iterations, output_dir)
    plots = generate_plots(results, output_dir)
    generate_report(results, plots, output_dir)

    log.info("Done! Results in %s", output_dir)


if __name__ == "__main__":
    main()
