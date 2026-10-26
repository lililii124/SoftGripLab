"""Command-line entry points; heavyweight backends are imported on demand."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="SoftGripLab mechanics benchmarks (SI units)")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Synthetic identification, grasp and shock-envelope benchmark")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--seed", type=int, default=7)
    fit = commands.add_parser("identify", help="Fit one or more uniformly sampled compression CSVs")
    fit.add_argument("csv", type=Path, nargs="+")
    fit.add_argument("--thickness-m", type=float, required=True)
    fit.add_argument("--area-m2", type=float, required=True)
    fit.add_argument("--noise-std-n", type=float, default=1.0)
    fit.add_argument("--output", type=Path, default=Path("outputs/material.json"))
    mpm = commands.add_parser("mpm-grasp", help="Optional 3D JAX-MPM squeeze/hold/base-shock scene")
    mpm.add_argument("--output", type=Path, default=Path("outputs/mpm"))
    mpm.add_argument("--grid", type=int, default=24)
    mpm.add_argument("--frames", type=int, default=40)
    mpm.add_argument("--friction", type=float, default=0.6)
    mpm.add_argument("--shock-peak", type=float, default=20.0, help="Vertical frame acceleration, m/s²")
    mpm.add_argument("--dt-factor", type=float, default=1.0)
    mpm.add_argument("--young-pa", type=float, default=35000)
    mpm.add_argument("--gif", action="store_true")
    compression = commands.add_parser("animate-compression", help="Animate saved compression records")
    compression.add_argument("--source", type=Path, default=Path("outputs/demo"))
    compression.add_argument("--output", type=Path, default=Path("outputs/gallery/01_compression.gif"))
    grasp = commands.add_parser("animate-grasp", help="Compare actual MPM trajectories as an animation")
    grasp.add_argument("--source", type=Path, default=Path("outputs/mpm"))
    grasp.add_argument("--comparison", type=Path, default=Path("outputs/mpm_frictionless"))
    grasp.add_argument("--output", type=Path, default=Path("outputs/gallery/03_grasp.gif"))
    adhesion = commands.add_parser("adhesion", help="Animate an upstream Hertz/JKR contact cycle")
    adhesion.add_argument("--output", type=Path, default=Path("outputs/adhesion"))
    adhesion.add_argument("--frames", type=int, default=151)
    robot = commands.add_parser("robot-demo", help="PyBullet Panda suction pick-and-place")
    robot.add_argument("--output", type=Path, default=Path("outputs/robot"))
    robot.add_argument("--pressure-pa", type=float, default=30000.)
    robot.add_argument("--no-render", action="store_true")
    robot.add_argument("--fps", type=int, default=15)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import run_demo
            result = run_demo(args.output, args.seed)
            print(json.dumps({"output": str(args.output), "fit": result["fit"],
                              "held_out": result["held_out"], "policies": result["policies"]}, indent=2))
        elif args.command == "identify":
            from .identification import CompressionTrace, fit_material
            from .materials import Geometry
            geometry = Geometry(args.thickness_m, args.area_m2)
            traces = [CompressionTrace.from_csv(path, geometry, args.noise_std_n) for path in args.csv]
            result = fit_material(traces).to_dict()
            result["input_files"] = [str(path) for path in args.csv]
            result["geometry"] = {"thickness_m": geometry.thickness_m, "area_m2": geometry.area_m2}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(json.dumps(result, indent=2))
        elif args.command == "animate-compression":
            from .animations import render_compression
            render_compression(args.source, args.output)
            print(args.output)
        elif args.command == "animate-grasp":
            from .animations import render_grasp
            render_grasp(args.source, args.comparison, args.output)
            print(args.output)
        elif args.command == "adhesion":
            from .backends.adhesive import run_adhesion
            print(json.dumps(run_adhesion(args.output, frames=args.frames), indent=2))
        elif args.command == "robot-demo":
            from .backends.robot import run_robot
            print(json.dumps(run_robot(args.output, pressure_pa=args.pressure_pa,
                                       render=not args.no_render, fps=args.fps), indent=2))
        else:
            from .backends.jax_mpm import run_grasp
            print(json.dumps(run_grasp(args.output, grid=args.grid, frames=args.frames,
                                       friction=args.friction, shock_peak=args.shock_peak,
                                       dt_factor=args.dt_factor, young_pa=args.young_pa,
                                       gif=args.gif), indent=2))
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        parser.exit(2, f"softgriplab: {error}\n")
