"""
Standard benchmark workload profiles for HBM thermal simulation.

Each benchmark returns per-bank read and write access counts for a single
sampling interval (1 ms).  The profiles are based on published DRAM-level
characterisation of standard GPU workloads on HBM2 systems.

Access counts are derived from bandwidth utilisation:

    Peak BW    = 256 GB/s  (8 ch × 128 bits × 2 Gbps)
    Per-channel= 32 GB/s
    Per-bank   = 32 / 16 = 2 GB/s  (16 banks per channel)
    Bytes/access = 64 B  (128-bit bus × BL4)
    Peak accesses/bank/ms = 2e9 * 1e-3 / 64 ≈ 31,250

Typical DRAM timing constraints (tRC ≈ 48 ns closed-page, tCCD ≈ 2 ns
open-page) further limit per-bank access rates.  Realistic workloads
sustain 40–80 % of the per-bank peak.
"""

import math
import random as _random

# Peak accesses per bank per 1 ms interval at 2 GHz HBM2
_PEAK_ACC_PER_BANK_PER_MS = 31_250


def _uniform_counts(num_banks, accesses_per_bank):
    """Every bank sees the same number of accesses."""
    return [int(accesses_per_bank)] * num_banks


def _gaussian_hotspot(num_banks, mean_accesses, sigma_fraction=0.3, seed=42):
    """Banks near the centre of each layer receive more traffic."""
    rng = _random.Random(seed)
    banks_per_layer = 16
    layers = num_banks // banks_per_layer
    counts = [0] * num_banks
    for z in range(layers):
        for b in range(banks_per_layer):
            row, col = divmod(b, 4)
            # Distance from centre (1.5, 1.5) in a 4×4 grid
            dist = math.sqrt((row - 1.5) ** 2 + (col - 1.5) ** 2)
            scale = math.exp(-0.5 * (dist / (sigma_fraction * 4)) ** 2)
            noise = rng.uniform(0.85, 1.15)
            counts[z * banks_per_layer + b] = int(mean_accesses * scale * noise)
    return counts


# ======================================================================
# Public benchmark generators
# ======================================================================

BENCHMARKS = {}


def _register(name):
    """Decorator to register a benchmark by name."""
    def decorator(fn):
        BENCHMARKS[name] = fn
        return fn
    return decorator


@_register("stream")
def stream_benchmark(cfg, step=0):
    """STREAM copy/triad: ~70 % peak BW, uniform across all banks.

    At 2 GHz HBM2, peak per-bank rate ≈ 31,250 accesses / ms.
    STREAM sustains ~70 % → ~21,875 accesses/ms/bank.
    Read:write ratio ≈ 2:1 (STREAM copy reads 2 arrays, writes 1).
    """
    total_per_bank = int(_PEAK_ACC_PER_BANK_PER_MS * 0.70)
    rd = int(total_per_bank * 2 / 3)
    wr = total_per_bank - rd
    reads = _uniform_counts(cfg.NUM_BANKS, rd)
    writes = _uniform_counts(cfg.NUM_BANKS, wr)
    return reads, writes


@_register("sgemm")
def sgemm_benchmark(cfg, step=0):
    """Dense SGEMM: ~45 % peak BW, bursty, concentrated on even channels.

    Matrix multiply has high arithmetic intensity; HBM traffic comes in
    bursts when tiles are loaded.  Even channels carry more traffic because
    of column-major tile layout.
    """
    base = int(_PEAK_ACC_PER_BANK_PER_MS * 0.45)
    reads = [0] * cfg.NUM_BANKS
    writes = [0] * cfg.NUM_BANKS
    banks_per_layer = cfg.banks_per_layer

    for b in range(cfg.NUM_BANKS):
        layer = b // banks_per_layer
        local = b % banks_per_layer
        channel = local // 2
        # Even channels get 1.5× traffic, odd channels get 0.5×
        factor = 1.5 if channel % 2 == 0 else 0.5
        # Time-varying bursty pattern
        burst = 1.2 if (step + layer) % 3 == 0 else 0.8
        rd = int(base * factor * burst * 0.8)
        wr = int(base * factor * burst * 0.2)
        reads[b] = rd
        writes[b] = wr
    return reads, writes


@_register("resnet50")
def resnet50_benchmark(cfg, step=0):
    """ResNet-50 inference: periodic burst–idle pattern.

    Convolution layers produce high BW bursts (~60 % peak); batch-norm
    and ReLU layers are compute-bound with low HBM traffic (~10 %).
    A full layer takes ~5 intervals; we model a 5-step repeating cycle.
    """
    phase = step % 5
    if phase < 3:
        # Convolution layer — high bandwidth
        bw_frac = 0.60
    else:
        # Batch-norm / ReLU — low bandwidth
        bw_frac = 0.10
    base = int(_PEAK_ACC_PER_BANK_PER_MS * bw_frac)
    # Read-heavy (weight loading + activation reads)
    rd = int(base * 0.85)
    wr = base - rd
    reads = _uniform_counts(cfg.NUM_BANKS, rd)
    writes = _uniform_counts(cfg.NUM_BANKS, wr)
    return reads, writes


@_register("random")
def random_benchmark(cfg, step=0):
    """Random / pointer-chasing: ~30 % peak BW, non-uniform across banks.

    Graph analytics and hash-table workloads generate random addresses.
    Each bank gets a random fraction of the mean traffic.
    """
    mean = int(_PEAK_ACC_PER_BANK_PER_MS * 0.30)
    rng = _random.Random(step * 1000 + 42)
    reads = [rng.randint(int(mean * 0.5), int(mean * 1.5)) for _ in range(cfg.NUM_BANKS)]
    writes = [rng.randint(int(mean * 0.1), int(mean * 0.4)) for _ in range(cfg.NUM_BANKS)]
    return reads, writes


@_register("hotspot_stress")
def hotspot_stress_benchmark(cfg, step=0):
    """Thermal stress test: heavy traffic on centre banks of upper layers.

    This creates a thermal hotspot in the middle of the stack to exercise
    DTM policies and demonstrate the temperature-gradient capability.
    """
    base = int(_PEAK_ACC_PER_BANK_PER_MS * 0.80)
    reads = _gaussian_hotspot(cfg.NUM_BANKS, int(base * 0.75), sigma_fraction=0.25)
    writes = _gaussian_hotspot(cfg.NUM_BANKS, int(base * 0.25), sigma_fraction=0.25)
    return reads, writes


def list_benchmarks():
    """Return sorted list of registered benchmark names."""
    return sorted(BENCHMARKS.keys())


def get_benchmark(name):
    """Return the benchmark function for *name*, or raise KeyError."""
    return BENCHMARKS[name]
