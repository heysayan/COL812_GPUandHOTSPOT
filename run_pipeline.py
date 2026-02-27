#!/usr/bin/env python3
"""
Entry point for the closed-loop AccelSim → HBM Power → HotSpot → DTM pipeline.

Usage examples
--------------

1. Run with a static per-bank access-count file::

     python run_pipeline.py --trace access_rates.txt --iterations 100

2. Run with per-interval trace files (one file per sampling interval)::

     python run_pipeline.py --trace-dir ./interval_traces/ --iterations 200

3. Run with a live Accel-Sim command::

     python run_pipeline.py \\
         --accelsim-cmd "./gpu-simulator/bin/release/accel-sim.out -config gpgpusim.config -trace kernelslist.g" \\
         --iterations 50

4. Customise thermal thresholds and output location::

     python run_pipeline.py --trace access_rates.txt \\
         --throttle-temp 78 --critical-temp 83 --cooldown-temp 73 \\
         --output-dir ./my_run/ --iterations 150
"""

import argparse
import logging
import sys
import os

# Ensure the pipeline package is importable from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import (
    PipelineConfig,
    PipelineRunner,
    ThrottlingPolicy,
    DVFSPolicy,
    LowPowerModePolicy,
    CompositeDTMPolicy,
)


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Closed-loop thermal management pipeline: "
                    "AccelSim → HBM Power Model → HotSpot → DTM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Trace source (mutually exclusive) ---
    src = p.add_argument_group("Trace source (pick one)")
    src.add_argument(
        "--trace", type=str, default=None,
        help="Path to a static per-bank access file (reads then writes, "
             "space-separated, one line each).",
    )
    src.add_argument(
        "--trace-dir", type=str, default=None,
        help="Directory with per-interval trace files named interval_<N>.txt.",
    )
    src.add_argument(
        "--accelsim-cmd", type=str, default=None,
        help="Shell command to execute Accel-Sim for each interval.",
    )

    # --- Pipeline control ---
    ctl = p.add_argument_group("Pipeline control")
    ctl.add_argument("--iterations", type=int, default=100,
                     help="Number of sampling intervals to simulate (default: 100).")
    ctl.add_argument("--output-dir", type=str, default=None,
                     help="Directory for output files.")
    ctl.add_argument("--board-config", type=str, default=None,
                     help="HotSpot board configuration name (default: 3Dmem_16core).")

    # --- Thermal thresholds ---
    therm = p.add_argument_group("Thermal thresholds (°C)")
    therm.add_argument("--throttle-temp", type=float, default=None)
    therm.add_argument("--critical-temp", type=float, default=None)
    therm.add_argument("--cooldown-temp", type=float, default=None)

    # --- DTM policy selection ---
    dtm = p.add_argument_group("DTM policy")
    dtm.add_argument(
        "--policy", type=str, default="composite",
        choices=["throttle", "dvfs", "lpm", "composite"],
        help="Which DTM policy to use (default: composite = all combined).",
    )

    # --- Logging ---
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable DEBUG-level logging.")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build configuration
    cfg = PipelineConfig(
        board_config=args.board_config,
        output_dir=args.output_dir,
    )
    if args.throttle_temp is not None:
        cfg.THROTTLE_TEMP_C = args.throttle_temp
    if args.critical_temp is not None:
        cfg.CRITICAL_TEMP_C = args.critical_temp
    if args.cooldown_temp is not None:
        cfg.COOLDOWN_TEMP_C = args.cooldown_temp

    # Build DTM policy
    if args.policy == "throttle":
        policy = ThrottlingPolicy(cfg)
    elif args.policy == "dvfs":
        policy = DVFSPolicy(cfg)
    elif args.policy == "lpm":
        policy = LowPowerModePolicy(cfg)
    else:  # composite
        policy = CompositeDTMPolicy(cfg, [
            ThrottlingPolicy(cfg),
            DVFSPolicy(cfg),
            LowPowerModePolicy(cfg),
        ])

    # Build and run the pipeline
    runner = PipelineRunner(cfg)
    runner.set_dtm_policy(policy)

    accelsim_cmd = args.accelsim_cmd.split() if args.accelsim_cmd else None

    history = runner.run(
        iterations=args.iterations,
        trace_source=args.trace,
        accelsim_cmd=accelsim_cmd,
        per_interval_trace_dir=args.trace_dir,
    )

    # Summary
    if history["bank_temps"]:
        import numpy as np
        all_temps = np.array(history["bank_temps"])
        print("\n=== Pipeline Summary ===")
        print("  Iterations completed : %d" % len(history["bank_temps"]))
        print("  Peak temperature     : %.2f °C" % np.max(all_temps))
        print("  Mean temperature     : %.2f °C" % np.mean(all_temps))
        print("  Final freq_scale     : %.2f" % history["freq_scale"][-1])
        print("  Final voltage_scale  : %.2f" % history["voltage_scale"][-1])
        print("  Output directory     : %s" % cfg.output_dir)
    else:
        print("\nNo temperature data collected (HotSpot may not be available).")
        print("  Power traces written to: %s" % cfg.output_dir)


if __name__ == "__main__":
    main()
