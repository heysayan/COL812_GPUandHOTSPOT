# Closed-Loop Thermal Management Pipeline for HBM on GPU

## Technical Report

### 1. Introduction

This document describes the design and implementation of an industry-grade
closed-loop thermal management pipeline for High Bandwidth Memory (HBM) in GPU
systems. The pipeline connects four stages in a feedback loop:

```
┌──────────┐     ┌───────────────┐     ┌─────────┐     ┌────────────┐
│ Accel-Sim│────▶│ HBM Power     │────▶│ HotSpot │────▶│ DTM Policy │
│ (trace)  │     │ Model         │     │ (temp)  │     │ (feedback) │
└──────────┘     └───────────────┘     └─────────┘     └─────┬──────┘
      ▲                                                       │
      └───────────────────────────────────────────────────────┘
                    throttle / DVFS / LPM adjustments
```

**Prior work.** The earlier `hotspot_tool/sample_script.py` was a toy project
that explored several DTM policies on synthetic workloads with a simplified
power model. It used hard-coded access rates and invoked HotSpot directly from
a monolithic script. While valuable for prototyping, it was not designed for
real benchmark workloads or production use.

**This work.** The new `pipeline/` package is a modular, testable, and
extensible framework that:

- Parses **real** Accel-Sim memory-access traces from benchmark simulations.
- Uses a **realistic** HBM power model accounting for dynamic read/write
  energy, refresh power, static leakage, and DVFS scaling.
- Invokes **HotSpot** for physics-based 3-D thermal simulation.
- Applies **pluggable DTM policies** — throttling, DVFS, low-power mode, and
  any user-defined combination — with feedback into the next simulation interval.

---

### 2. Architecture Overview

The pipeline is implemented as a Python package at the repository root, with
the following module structure:

```
pipeline/
├── __init__.py                  # Public API exports
├── config.py                    # PipelineConfig — all parameters in one place
├── accelsim_trace_parser.py     # AccelSimTraceParser — read AccelSim output
├── hbm_power_model.py           # HBMPowerModel — access counts → power trace
├── hotspot_interface.py         # HotSpotInterface — invoke HotSpot, read temps
├── dtm_policies.py              # DTM policies (Throttle, DVFS, LPM, Composite)
├── pipeline_runner.py           # PipelineRunner — closed-loop orchestrator
└── tests/
    └── test_pipeline.py         # 28 unit tests

run_pipeline.py                  # CLI entry point
```

The package lives at the repository root (not inside `hotspot_tool/`) because
it spans both the Accel-Sim GPU simulator and the HotSpot thermal tool.

---

### 3. Module Descriptions

#### 3.1 `config.py` — PipelineConfig

Centralizes every tunable parameter of the pipeline:

| Category | Parameters |
|----------|-----------|
| **HBM Geometry** | 128 banks (4×4×8), 16 logic cores, 8 channels, 16 pseudo-channels |
| **Energy** | Read/write energy per access (20.55 pJ), refresh energy (3.55 pJ), static leakage |
| **Frequency** | Nominal 2 GHz, range 0.5–3.2 GHz; nominal voltage 1.1 V, range 0.8–1.2 V |
| **Timing** | 1 ms sampling interval, 7.8 µs refresh interval (t_REFI) |
| **Thermal thresholds** | Throttle at 80°C, critical at 85°C, cooldown at 75°C |
| **DVFS steps** | 4 threshold/factor pairs for frequency and voltage scaling |
| **Paths** | HotSpot binary, config files, floorplans, output directory |

All values can be overridden at construction time or via the CLI.

#### 3.2 `accelsim_trace_parser.py` — AccelSimTraceParser

Extracts per-bank HBM read and write counts from Accel-Sim output. Supports
three input formats:

1. **Per-bank regex** — lines like `dram[0]: bk[3]: rd = 42`
2. **Bulk CSV** — lines like `dram_reads_per_bank = 100,200,...`
3. **Raw access file** — two lines of space-separated per-bank counts (the
   simple format used by the legacy `sample_script.py`)

Also extracts `gpu_sim_cycle` for timing correlation.

#### 3.3 `hbm_power_model.py` — HBMPowerModel

Converts per-bank access counts into a power trace compatible with HotSpot.

**Power calculation per bank:**

```
P_dynamic = (N_rd × E_rd + N_wr × E_wr) / Δt × V²_scale × f_scale
P_refresh = (Δt / t_REFI) × rows_per_refresh × E_refresh / Δt
P_static  = P_leakage × V_scale
P_total   = P_dynamic + P_refresh + P_static       (if bank is active)
P_total   = P_leakage × V_scale × 0.1              (if bank is in LPM)
```

The DVFS-aware scaling follows the standard CMOS relationship `P ∝ C·V²·f`.

The model also generates the HotSpot ptrace header (LC_0…LC_15, B_0…B_127)
and writes properly formatted power trace files.

#### 3.4 `hotspot_interface.py` — HotSpotInterface

Manages the HotSpot tool lifecycle:

- **Setup** — copies initial temperature file, creates output trace files.
- **Run** — builds the full HotSpot command line with all flags (`-c`, `-p`,
  `-type 3Dmem`, `-detailed_3D on`, `-model_type grid`, per-layer voltages,
  etc.) and invokes it as a subprocess.
- **Temperature readers** — parses the temperature trace file and provides
  per-bank, per-channel, and per-pseudo-channel peak temperatures.
- **State propagation** — copies the transient output to the init file for the
  next iteration, and appends data to the full history trace files.

Gracefully handles the case where the HotSpot binary is not available (returns
`False` so the pipeline can still generate power traces).

#### 3.5 `dtm_policies.py` — DTM Policies

All policies inherit from the abstract `DTMPolicy` base class and return a
`DTMAction` object containing:

| Field | Type | Description |
|-------|------|-------------|
| `throttle_factors` | `list[float]` | Per-bank multiplicative throttle (1.0 = full, 0.0 = blocked) |
| `freq_scale` | `float` | Global frequency scale factor |
| `voltage_scale` | `float` | Global voltage scale factor |
| `bank_active` | `list[bool]` | Per-bank active/LPM flag |
| `lpm_channels` | `set[int]` | Channels currently in low-power mode |
| `lpm_pseudochannels` | `set[int]` | Pseudo-channels in low-power mode |
| `migration_pairs` | `list[tuple]` | Channel migration pairs |

**Implemented policies:**

| Policy | Description |
|--------|-------------|
| `ThrottlingPolicy` | Linearly reduces per-bank access rates between `throttle_temp` and `critical_temp`. At critical temperature, accesses are fully blocked. |
| `DVFSPolicy` | Step-wise global frequency and voltage scaling based on peak temperature. Four configurable threshold/factor pairs. |
| `LowPowerModePolicy` | Gates entire channels into low-power mode when peak temperature exceeds `critical_temp`. Restores when below `cooldown_temp`. |
| `CompositeDTMPolicy` | Chains multiple policies in sequence; each refines the action of the previous one. Default configuration applies all three policies. |

#### 3.6 `pipeline_runner.py` — PipelineRunner

The main orchestrator that runs the closed loop:

```python
for step in range(iterations):
    # Stage 1: Get access trace
    reads, writes = get_access_trace(step)
    reads, writes = apply_throttle(reads, writes)  # DTM feedback

    # Stage 2: Compute power
    bank_power, logic_power = power_model.compute(reads, writes, bank_active,
                                                   freq_scale, voltage_scale)
    write_power_trace(bank_power, logic_power)

    # Stage 3: Run HotSpot
    hotspot.run(voltage_per_layer)
    bank_temps = hotspot.get_bank_temperatures()

    # Stage 4: DTM policy
    action = dtm_policy.evaluate(bank_temps)
    apply_action(action)  # Updates throttle, DVFS, LPM for next iteration
```

**Trace sources** (mutually exclusive):

1. **Static file** — a single access-count file reused every interval.
2. **Per-interval directory** — files named `interval_0.txt`, `interval_1.txt`, etc.
3. **Live Accel-Sim** — a subprocess command whose stdout is parsed.
4. **Idle fallback** — zero accesses (for testing).

The runner records full history for post-analysis: per-step temperatures,
power, frequency/voltage scale, LPM state, and throttle factors.

#### 3.7 `run_pipeline.py` — CLI Entry Point

Provides a command-line interface with argument groups for trace source,
pipeline control, thermal thresholds, and DTM policy selection. Example:

```bash
python run_pipeline.py \
    --trace access_rates.txt \
    --iterations 200 \
    --policy composite \
    --throttle-temp 78 \
    --critical-temp 83 \
    --output-dir ./results/
```

---

### 4. HBM Power Model Details

The power model (`pipeline/hbm_power_model.py`) computes per-bank power from
five physical components:

#### 4.1 Dynamic Read/Write Power

Energy per DRAM access from CACTI3DD modelling of HBM2 bank:
- **Energy per read:** 20.55 nJ (includes row-buffer sense-amp, column decode,
  bit-line charge, I/O driver)
- **Energy per write:** 20.55 nJ
- **P_dynamic = (reads × E_rd + writes × E_wr) / Δt × V²·f**

#### 4.2 Activation / Precharge Power

Each read/write implies a row activation (open) and precharge (close):
- **Activation energy:** 3.2 nJ per row-open
- **Precharge energy:** 1.1 nJ per row-close
- **P_act_pre = accesses × (E_act + E_pre) / Δt × V²·f**

#### 4.3 Refresh Power

DRAM cells require periodic refresh at t_REFI = 7.8 µs:
- **Refresh energy:** 3.55 nJ per refresh command (from CACTI3DD)
- **P_refresh = avg_refreshes_per_interval × E_ref / Δt** (constant background)

#### 4.4 Temperature-Dependent Static (Leakage) Power

Sub-threshold leakage scales exponentially with temperature:

    P_leak(T) = P_ref × exp(α × (T − T_ref))

Parameters (20 nm MOSFET technology):
- **P_ref:** 20 mW per bank at T_ref = 45°C
- **α:** 0.06 /°C → leakage doubles every ~11.5°C
- In low-power mode: 10% of leakage retained (clock gating + power gating)
- Scales linearly with voltage under DVFS

This creates a thermal-leakage feedback loop: higher temperature → more
leakage → more power → higher temperature. The pipeline feeds HotSpot
temperatures back into the power model each iteration.

#### 4.5 Logic Core Power

Each of 16 logic cores (I/O PHY, DLL, command decoder, ECC engine):
- **Static:** 20 mW per core
- **Dynamic:** 15 mW per core (data routing, ECC), scaled by V²·f

Total logic die power at nominal: 16 × 35 mW = 560 mW.

#### 4.6 DVFS Scaling

Dynamic power scales as V²·f (standard CMOS). Static/leakage scales linearly
with voltage.

---

### 5. Standard Benchmark Workloads

The pipeline includes five standard benchmark workload profiles
(`pipeline/benchmarks.py`) based on published HBM2 bandwidth utilisation:

| Benchmark | BW Utilisation | Pattern | Use Case |
|-----------|---------------|---------|----------|
| `stream` | 70% peak | Uniform, sustained | Worst-case thermal |
| `sgemm` | 45% peak | Bursty, channel-skewed | Dense linear algebra |
| `resnet50` | 60%/10% alternating | Burst–idle cycle | DNN inference |
| `random` | 30% peak | Non-uniform random | Graph analytics |
| `hotspot_stress` | 80% peak | Centre-concentrated | Thermal stress test |

Access counts are derived from HBM2 bandwidth at 2 GHz:
- Peak per-bank rate: 31,250 accesses/ms (2 GB/s / 64 B per access)
- Each "access" = one DRAM command transferring 64 bytes (128-bit bus × BL4)

Run benchmarks with: `python run_benchmark.py --benchmark stream --iterations 50`

---

### 6. DTM Policy Design

The DTM policies are designed to be composable. The `CompositeDTMPolicy` chains
policies in order:

1. **ThrottlingPolicy** runs first: applies per-bank access-rate reduction
   proportional to how far the bank temperature exceeds the throttle threshold.
2. **DVFSPolicy** runs second: selects the appropriate global frequency/voltage
   operating point based on peak temperature across all banks.
3. **LowPowerModePolicy** runs last: as a safety net, completely gates channels
   whose temperatures exceed the critical threshold.

This layered approach ensures:
- Fine-grained control at the bank level (throttling)
- System-wide efficiency optimization (DVFS)
- Fail-safe thermal protection (LPM)

---

### 6. Integration with Accel-Sim

The pipeline integrates with Accel-Sim in three ways:

1. **Offline trace parsing** — Post-process Accel-Sim output files to extract
   `dram_reads_per_bank` and `dram_writes_per_bank` statistics.
2. **Per-interval traces** — When Accel-Sim is configured to dump stats at
   regular intervals, each interval's stats file is read sequentially.
3. **Live subprocess** — The pipeline can launch Accel-Sim as a child process,
   capture its stdout, and parse access statistics in real time.

The DTM feedback affects the next interval:
- **Throttled banks** have their read/write counts reduced proportionally.
- **DVFS** changes the effective frequency, which scales the dynamic power.
- **LPM banks** are treated as inactive, producing near-zero power.

---

### 7. Integration with HotSpot

The HotSpot integration uses the 3D grid model with the real HBM configuration
files from `config/hotspot/3Dmem_16core/`:

- **Stack type:** `3Dmem` (3D memory stack with logic layer)
- **Model type:** Grid (8×8 spatial resolution)
- **Detailed 3D:** Enabled
- **Leakage-temperature loop:** Enabled (`-leakage_used 1`)
- **Sampling interval:** 1000 ms (matching `mem_hotspot.config`)
- **Ambient temperature:** 318.15 K (45°C)
- **Thermal threshold:** 354.95 K (81.8°C)

#### 7.1 Layer Structure (18 layers)

The `mem.lcf` defines the full 3D HBM stack:

| Layer | Type | Power? | Thickness | Floorplan |
|-------|------|--------|-----------|-----------|
| 0 | mem_ctrl (logic cores) | Yes | 50 µm | `mem_ctrl.flp` (LC_0..LC_15, 4×4 grid) |
| 1 | TIM | No | 20 µm | `mem_tim.flp` (TB_0..TB_15) |
| 2 | mem_bank_1 | Yes | 50 µm | `mem_bank_1.flp` (B_0..B_15, 4×4 grid) |
| 3 | TIM | No | 20 µm | `mem_tim.flp` |
| 4 | mem_bank_2 | Yes | 50 µm | `mem_bank_2.flp` |
| ... | ... | ... | ... | ... |
| 16 | mem_bank_8 | Yes | 50 µm | `mem_bank_8.flp` |
| 17 | TIM | No | 20 µm | `mem_tim.flp` |

Total: 1 logic layer + 8 bank layers + 9 TIM layers = 18 layers.

#### 7.2 Floorplan Geometry

Each bank layer has 16 banks in a 4×4 grid (each 1.707×1.707 mm).
The logic layer has 16 cores in the same 4×4 grid.
Total die size: 6.828×6.828 mm.
Generated by: `floorplanlib/create.py --mode 3Dmem --cores 4x4 --banks 4x4x8`.

#### 7.3 Path Resolution

The shipped `mem.lcf` stores floorplan paths relative to the repository root.
At runtime, `PipelineConfig.prepare_lcf()` resolves these to absolute paths and
writes a runtime copy to the output directory.  This allows the same config
files to work on any machine without modification.

#### 7.4 Thermal Model Parameters (from `mem_hotspot.config`)

| Parameter | Value |
|-----------|-------|
| Chip thickness | 100 µm |
| Silicon conductivity | 100 W/(m·K) |
| Silicon specific heat | 1.75×10⁶ J/(m³·K) |
| Heatsink side | 46.828 mm |
| Spreader side | 26.828 mm |
| Convection resistance | 0.1 K/W |

---

### 9. Testing

The test suite (`pipeline/tests/test_pipeline.py`) contains 49 tests
covering all modules:

| Module | Tests | Description |
|--------|-------|-------------|
| `PipelineConfig` | 4 | Default values, bank/channel mapping |
| `AccelSimTraceParser` | 5 | All three input formats, missing files |
| `HBMPowerModel` | 5 | Idle power, LPM vs active, DVFS scaling, file I/O |
| `ThrottlingPolicy` | 3 | Below threshold, above critical, partial throttle |
| `DVFSPolicy` | 2 | Cool (no scaling), hot (triggers scaling) |
| `LowPowerModePolicy` | 3 | LPM triggered, cool (no LPM), cooldown restores |
| `CompositeDTMPolicy` | 1 | Chained policy evaluation |
| `PipelineRunner` | 3 | Static trace, per-interval dir, idle run |
| `HotSpotInterface` | 2 | Setup creates files, missing temp file returns zeros |
| `RealConfigFiles` | 9 | Config/floorplan existence, LCF layers, path resolution |
| `RealisticPowerModel` | 6 | Nonzero idle power, temp-dependent leakage, act/pre energy |
| `Benchmarks` | 6 | All 5 benchmarks, realistic counts, pipeline integration |

All 49 tests pass.

---

### 10. File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `pipeline/__init__.py` | 38 | Package init, public API |
| `pipeline/config.py` | 197 | Configuration parameters |
| `pipeline/accelsim_trace_parser.py` | 150 | Accel-Sim trace parsing |
| `pipeline/hbm_power_model.py` | 165 | Realistic HBM power model |
| `pipeline/hotspot_interface.py` | 190 | HotSpot interface |
| `pipeline/dtm_policies.py` | 193 | DTM policies |
| `pipeline/pipeline_runner.py` | 230 | Pipeline orchestrator |
| `pipeline/benchmarks.py` | 155 | Standard benchmark workloads |
| `pipeline/tests/test_pipeline.py` | ~560 | Unit tests (49 tests) |
| `run_pipeline.py` | 164 | CLI entry point |
| `run_benchmark.py` | ~470 | Benchmark simulation + plotting |
| `REPORT.md` | — | This document |

---

### 11. Usage Guide

#### Prerequisites

- Python 3.8+
- NumPy (`pip install numpy`)
- HotSpot binary compiled in `hotspot_tool/` (`cd hotspot_tool && make`)
- Accel-Sim traces (from `get-accel-sim-traces.py` or your own benchmarks)
- HotSpot board config files in `config/hotspot/<board_config>/`

#### Running the Pipeline

**With a static trace file:**
```bash
python run_pipeline.py --trace access_rates.txt --iterations 200
```

**With per-interval traces from Accel-Sim:**
```bash
python run_pipeline.py --trace-dir ./interval_traces/ --iterations 500
```

**With live Accel-Sim execution:**
```bash
python run_pipeline.py \
    --accelsim-cmd "./gpu-simulator/bin/release/accel-sim.out -config gpgpusim.config -trace kernelslist.g" \
    --iterations 100
```

**Custom thermal thresholds and policy:**
```bash
python run_pipeline.py \
    --trace access_rates.txt \
    --policy composite \
    --throttle-temp 78 --critical-temp 83 --cooldown-temp 73 \
    --output-dir ./results/ \
    --iterations 300 -v
```

#### Programmatic API

```python
from pipeline import (
    PipelineConfig, PipelineRunner,
    ThrottlingPolicy, DVFSPolicy, LowPowerModePolicy, CompositeDTMPolicy,
)

cfg = PipelineConfig(output_dir="./my_run")
cfg.THROTTLE_TEMP_C = 78.0
cfg.CRITICAL_TEMP_C = 83.0

policy = CompositeDTMPolicy(cfg, [
    ThrottlingPolicy(cfg),
    DVFSPolicy(cfg),
    LowPowerModePolicy(cfg),
])

runner = PipelineRunner(cfg)
runner.set_dtm_policy(policy)
history = runner.run(iterations=200, trace_source="access_rates.txt")

# Analyze results
import numpy as np
temps = np.array(history["bank_temps"])
print("Peak temperature:", np.max(temps), "°C")
```

#### Running Standard Benchmarks

```bash
# List available benchmarks
python run_benchmark.py --list

# Run STREAM benchmark with thermal plots
python run_benchmark.py --benchmark stream --iterations 50 --output-dir results/stream

# Run SGEMM benchmark
python run_benchmark.py --benchmark sgemm --iterations 100

# Run ResNet-50 inference pattern
python run_benchmark.py --benchmark resnet50 --iterations 100
```

Each benchmark run produces:
- `results.json` — raw simulation data
- `thermal_trace.png` — peak/mean temperature over time
- `power_trace.png` — total stack power over time
- `layer_heatmaps.png` — per-layer 4×4 thermal heatmap
- `vertical_profile.png` — temperature vs. layer depth
- `layer_power_dist.png` — power distribution across layers
- `ANALYSIS_REPORT.md` — complete analysis report with embedded plots

#### Programmatic Benchmark Usage

```python
from pipeline import PipelineConfig, PipelineRunner

cfg = PipelineConfig(output_dir="./my_run")
runner = PipelineRunner(cfg)
history = runner.run(iterations=50, benchmark="stream")
```

---

### 12. Comparison with Prior Work (sample_script.py)

| Aspect | sample_script.py (toy) | pipeline/ (industry-grade) |
|--------|----------------------|---------------------------|
| **Workload** | Synthetic random access rates | Real Accel-Sim benchmark traces + 5 standard benchmarks |
| **Power model** | Simple `E × N / Δt`, static = 0 | 5-component model: dynamic, act/pre, refresh, temp-dependent leakage, logic |
| **Leakage** | Constant zero | Exponential: P(T) = P_ref · exp(α·(T−T_ref)), doubles every ~11.5°C |
| **DTM policies** | 6 monolithic functions (dtm1–dtm6) | Modular, composable policy classes |
| **DVFS** | Fixed voltage passed to HotSpot | Dynamic voltage/frequency scaling with configurable thresholds |
| **Throttling** | Binary (on/off via bankstate) | Continuous per-bank throttle factor (0.0–1.0) |
| **Benchmarks** | None (synthetic only) | STREAM, SGEMM, ResNet-50, random, hotspot_stress |
| **Analysis** | None | Auto-generated plots + markdown report |
| **Configuration** | Hard-coded globals | Centralized `PipelineConfig` with CLI overrides |
| **Testing** | None | 49 unit tests |
| **Modularity** | Single 1100-line script | 8 focused modules (~1800 lines total) |
| **Error handling** | `os.system()` calls | `subprocess.run()` with error handling and graceful degradation |
| **Extensibility** | Requires editing the script | Add new `DTMPolicy` subclass; plug into composite |

---

### 12. Future Work

- **Per-channel DVFS** — extend DVFSPolicy to apply different frequency/voltage
  per HBM channel rather than globally.
- **Channel migration** — implement data-migration-aware DTM that moves hot
  channel contents to cooler channels (scaffolding exists in `DTMAction.migration_pairs`).
- **Accel-Sim integration tightening** — modify Accel-Sim's simulation loop to
  call the DTM policy between kernel intervals, enabling true co-simulation.
- **Visualization** — add plotting utilities to generate thermal maps and time
  series from the history data (can build on existing `visualize.py`).
- **Leakage-temperature feedback** — enable HotSpot's built-in
  temperature-leakage loop for more accurate power estimation.
