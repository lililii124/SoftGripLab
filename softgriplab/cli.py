"""Command-line entry points; heavyweight backends are imported on demand."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SoftGripLab mechanics benchmarks (SI units)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Synthetic identification, grasp and shock-envelope benchmark"
    )
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--seed", type=int, default=7)
    active = commands.add_parser(
        "active-identification", help="Axial adhesion: ambiguity, active probing and release prediction"
    )
    active.add_argument("--output", type=Path, default=Path("outputs/active_identification"))
    active.add_argument("--hypotheses", type=int, default=512)
    active.add_argument("--trials", type=int, default=6)
    active.add_argument("--rounds", type=int, default=2)
    active.add_argument("--seed", type=int, default=7)
    active.add_argument("--dt", type=float, default=0.0001)
    active.add_argument("--no-render", action="store_true")
    shape = commands.add_parser(
        "shape-identification", help="3D hyperelastic/adhesion identification on a curved lobed specimen"
    )
    shape.add_argument("--output", type=Path, default=Path("outputs/shape_identification"))
    shape.add_argument("--library", type=Path)
    shape.add_argument("--library-count", type=int, default=13)
    shape.add_argument("--materials", type=int, default=6)
    shape.add_argument("--trials", type=int, default=24)
    shape.add_argument("--seed", type=int, default=41)
    shape.add_argument("--no-dense", action="store_true")
    shape.add_argument("--no-render", action="store_true")
    shape_replay = commands.add_parser("animate-shape-identification", help="Replay saved 3D identification meshes")
    shape_replay.add_argument("--source", type=Path, default=Path("outputs/shape_identification"))
    unknown = commands.add_parser("unknown-surface-identification", help="Partial point cloud to geometry and material identification")
    unknown.add_argument("--output", type=Path, default=Path("outputs/unknown_surface"))
    unknown.add_argument("--library-count", type=int, default=13)
    unknown.add_argument("--trials", type=int, default=12)
    unknown.add_argument("--seed", type=int, default=73)
    unknown.add_argument("--no-dense", action="store_true")
    unknown.add_argument("--no-render", action="store_true")
    reconstruct = commands.add_parser("reconstruct-surface", help="Reconstruct a registered partial xyz scan under a swept-ellipse prior")
    reconstruct.add_argument("csv", type=Path)
    reconstruct.add_argument("--output", type=Path, default=Path("outputs/reconstructed_surface"))
    reconstruct.add_argument("--sigma-m", type=float, default=4e-5)
    fit = commands.add_parser(
        "identify", help="Fit one or more uniformly sampled compression CSVs"
    )
    fit.add_argument("csv", type=Path, nargs="+")
    fit.add_argument("--thickness-m", type=float, required=True)
    fit.add_argument("--area-m2", type=float, required=True)
    fit.add_argument("--noise-std-n", type=float, default=1.0)
    fit.add_argument("--output", type=Path, default=Path("outputs/material.json"))
    mpm = commands.add_parser(
        "mpm-grasp", help="Optional 3D JAX-MPM squeeze/hold/base-shock scene"
    )
    mpm.add_argument("--output", type=Path, default=Path("outputs/mpm"))
    mpm.add_argument("--grid", type=int, default=32)
    mpm.add_argument("--frames", type=int, default=40)
    mpm.add_argument("--friction", type=float, default=0.6)
    mpm.add_argument(
        "--shock-peak",
        type=float,
        default=20.0,
        help="Vertical frame acceleration, m/s²",
    )
    mpm.add_argument("--dt-factor", type=float, default=1.0)
    mpm.add_argument("--young-pa", type=float, default=35000)
    mpm.add_argument("--gif", action="store_true")
    compression = commands.add_parser(
        "animate-compression", help="Render saved FEM compression mesh"
    )
    compression.add_argument(
        "--source", type=Path, default=Path("outputs/fem_compression")
    )
    grasp = commands.add_parser(
        "animate-grasp", help="Compare actual MPM trajectories as an animation"
    )
    grasp.add_argument("--source", type=Path, default=Path("outputs/mpm"))
    grasp.add_argument(
        "--comparison", type=Path, default=Path("outputs/mpm_frictionless")
    )
    grasp.add_argument(
        "--output", type=Path, default=Path("outputs/gallery/03_grasp.gif")
    )
    adhesion = commands.add_parser(
        "adhesion", help="Export an upstream Hertz/JKR reference cycle (CSV)"
    )
    adhesion.add_argument("--output", type=Path, default=Path("outputs/adhesion"))
    adhesion.add_argument("--frames", type=int, default=151)
    fem = commands.add_parser(
        "fem-demo", help="Solve and render a tetrahedral deformable specimen"
    )
    fem.add_argument("--scene", choices=["compression", "grasp"], default="compression")
    fem.add_argument("--output", type=Path, default=Path("outputs/fem_compression"))
    fem.add_argument("--cells", type=int, default=12)
    fem.add_argument("--frames", type=int, default=91)
    fem.add_argument("--dt", type=float, default=0.0005)
    fem.add_argument("--friction", type=float, default=0.7)
    fem.add_argument("--no-render", action="store_true")
    continuum = commands.add_parser(
        "mpm-adhesion", help="Solve actual deformable adhesive sphere contact"
    )
    continuum.add_argument(
        "--output", type=Path, default=Path("outputs/continuum_adhesion")
    )
    continuum.add_argument("--grid", type=int, default=36)
    continuum.add_argument("--frames", type=int, default=101)
    continuum.add_argument("--work", type=float, default=14.4)
    continuum.add_argument("--dt-factor", type=float, default=1.0)
    comparison = commands.add_parser(
        "animate-adhesion", help="Render two saved continuum trajectories"
    )
    comparison.add_argument(
        "--source", type=Path, default=Path("outputs/continuum_adhesion")
    )
    comparison.add_argument(
        "--comparison", type=Path, default=Path("outputs/continuum_nonadhesive")
    )
    comparison.add_argument(
        "--output", type=Path, default=Path("outputs/gallery/02_adhesion.gif")
    )
    robot = commands.add_parser(
        "robot-demo", help="PyBullet Panda frictional pick-and-place"
    )
    robot.add_argument("--output", type=Path, default=Path("outputs/robot"))
    robot.add_argument("--friction", type=float, default=0.8)
    robot.add_argument("--finger-force-n", type=float, default=4.0)
    robot.add_argument("--no-render", action="store_true")
    robot.add_argument("--fps", type=int, default=15)
    coupled = commands.add_parser(
        "adhesive-robot", help="Panda and deformable payload coupled by adhesive forces"
    )
    coupled.add_argument("--output", type=Path, default=Path("outputs/adhesive_robot"))
    coupled.add_argument("--work", type=float, default=4.0)
    coupled.add_argument("--grid", type=int, default=40)
    coupled.add_argument(
        "--release-mode", choices=("inertial", "switch"), default="inertial"
    )
    coupled.add_argument("--release-duration", type=float, default=0.04)
    coupled.add_argument("--release-gain", type=float, default=0.6)
    coupled.add_argument("--dt-factor", type=float, default=0.5)
    coupled.add_argument("--coupling-dt", type=float, default=1 / 120)
    coupled.add_argument("--fps", type=int, default=15)
    coupled.add_argument("--no-render", action="store_true")
    replay = commands.add_parser(
        "animate-adhesive-robot", help="Replay coupled robot and material states"
    )
    replay.add_argument("--source", type=Path, default=Path("outputs/adhesive_robot"))
    cutting = commands.add_parser(
        "banana-cutting-disect",
        help="Three sequential FEM cuts with the optional DiSECt research backend",
    )
    cutting.add_argument("--output", type=Path, default=Path("outputs/banana_robot"))
    cutting.add_argument("--backend", type=Path)
    cutting.add_argument("--dt", type=float, default=2e-5)
    cutting.add_argument("--coupling-dt", type=float, default=1 / 120)
    cutting.add_argument("--duration", type=float, default=4.2)
    cutting.add_argument("--softening-rate", type=float, default=2.5e7)
    cutting.add_argument("--fps", type=int, default=15)
    cutting.add_argument("--no-render", action="store_true")
    pd = commands.add_parser(
        "banana-cutting", help="PeriPy cutting with no prescribed interfaces"
    )
    pd.add_argument("--output", type=Path, default=Path("outputs/banana_robot"))
    pd.add_argument("--backend", type=Path)
    pd.add_argument("--spacing", type=float, default=0.0025)
    pd.add_argument("--dt", type=float, default=5e-5)
    pd.add_argument("--coupling-dt", type=float, default=1 / 120)
    pd.add_argument("--duration", type=float, default=4.2)
    pd.add_argument("--critical-stretch", type=float, default=0.5)
    pd.add_argument("--no-knife-severing", action="store_true")
    pd.add_argument("--cut-x", type=float, nargs="+", default=(0.045, 0.0, -0.045))
    pd.add_argument("--fps", type=int, default=20)
    pd.add_argument("--no-render", action="store_true")
    cut_replay = commands.add_parser(
        "animate-cutting", help="Replay saved cutting data without importing the solver"
    )
    cut_replay.add_argument("--source", type=Path, default=Path("outputs/banana_robot"))
    args = parser.parse_args(argv)
    try:
        if args.command == "reconstruct-surface":
            import numpy as np
            from .surface_geometry import ScannedContact, reconstruct_surface
            from .complex_contact import LENGTH_M
            from .meshes import write_mesh

            points = np.loadtxt(args.csv, delimiter=",", skiprows=1, ndmin=2)
            geometry, report = reconstruct_surface(points, sigma_m=args.sigma_m)
            model = ScannedContact(geometry, (20,8,8))
            args.output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output/"geometry.npz", stations=geometry.stations, values=geometry.values,
                                input_points_m=points, length_scale_m=LENGTH_M)
            write_mesh(args.output, model.points*LENGTH_M, model.tets, model.faces)
            (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
            print(json.dumps({"output": str(args.output), **report}, indent=2))
        elif args.command == "unknown-surface-identification":
            from .unknown_surface import run_unknown_surface

            result = run_unknown_surface(args.output, library_count=args.library_count, trials=args.trials,
                                         seed=args.seed, dense=not args.no_dense, render=not args.no_render)
            print(json.dumps({"output": str(args.output), "surface":result["surface"],
                              "summary":result["summary"], "dense_validation":result["dense_validation"]}, indent=2))
        elif args.command == "shape-identification":
            from .shape_identification import run_shape_identification

            if args.materials < 2 or args.trials < 1 or args.library_count < 5:
                raise ValueError("materials >= 2, trials >= 1, library-count >= 5 are required")
            result = run_shape_identification(
                args.output, library_count=args.library_count, trials=args.trials,
                seed=args.seed, render=not args.no_render, library_path=args.library,
                validation_materials=args.materials, dense=not args.no_dense
            )
            print(json.dumps({"output": str(args.output), "summary": result["summary"],
                              "dense_validation": result["dense_validation"]}, indent=2))
        elif args.command == "animate-shape-identification":
            from .shape_render import render_shape_identification

            render_shape_identification(args.source)
            print(args.source/"shape_identification.gif")
        elif args.command == "active-identification":
            from .active_identification import run_active_identification

            report = run_active_identification(
                args.output, hypotheses=args.hypotheses, trials=args.trials,
                rounds=args.rounds, seed=args.seed, dt=args.dt, render=not args.no_render
            )
            print(json.dumps({"output": str(args.output), "policies": report["policies"],
                              "ambiguity_pair": report["ambiguity_pair"],
                              "integration_checks": report["integration_checks"]}, indent=2))
        elif args.command == "demo":
            from .demo import run_demo

            result = run_demo(args.output, args.seed)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "fit": result["fit"],
                        "held_out": result["held_out"],
                        "policies": result["policies"],
                    },
                    indent=2,
                )
            )
        elif args.command == "identify":
            from .identification import CompressionTrace, fit_material
            from .materials import Geometry

            geometry = Geometry(args.thickness_m, args.area_m2)
            traces = [
                CompressionTrace.from_csv(path, geometry, args.noise_std_n)
                for path in args.csv
            ]
            result = fit_material(traces).to_dict()
            result["input_files"] = [str(path) for path in args.csv]
            result["geometry"] = {
                "thickness_m": geometry.thickness_m,
                "area_m2": geometry.area_m2,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            print(json.dumps(result, indent=2))
        elif args.command == "animate-compression":
            from .mesh_render import render_fem

            render_fem(args.source)
            print(args.source)
        elif args.command == "fem-demo":
            from .backends.fem import run_fem

            print(
                json.dumps(
                    run_fem(
                        args.output,
                        scene=args.scene,
                        cells=args.cells,
                        frames=args.frames,
                        timestep_s=args.dt,
                        friction=args.friction,
                        render=not args.no_render,
                    ),
                    indent=2,
                )
            )
        elif args.command == "mpm-adhesion":
            from .backends.continuum_adhesion import run_continuum_adhesion

            print(
                json.dumps(
                    run_continuum_adhesion(
                        args.output,
                        grid=args.grid,
                        frames=args.frames,
                        work_j_m2=args.work,
                        dt_factor=args.dt_factor,
                    ),
                    indent=2,
                )
            )
        elif args.command == "animate-adhesion":
            from .mesh_render import render_adhesion_pair

            render_adhesion_pair(args.source, args.comparison, args.output)
            print(args.output)
        elif args.command == "animate-grasp":
            from .animations import render_grasp

            render_grasp(args.source, args.comparison, args.output)
            print(args.output)
        elif args.command == "adhesion":
            from .backends.adhesive import run_adhesion

            print(json.dumps(run_adhesion(args.output, frames=args.frames), indent=2))
        elif args.command == "animate-adhesive-robot":
            from .advanced_render import render_adhesive_robot

            render_adhesive_robot(args.source)
            print(args.source / "adhesive_robot.gif")
        elif args.command == "adhesive-robot":
            from .backends.adhesive_robot import run_adhesive_robot

            print(
                json.dumps(
                    run_adhesive_robot(
                        args.output,
                        work_j_m2=args.work,
                        grid=args.grid,
                        dt_factor=args.dt_factor,
                        coupling_dt=args.coupling_dt,
                        fps=args.fps,
                        render=not args.no_render,
                        release_mode=args.release_mode,
                        release_duration_s=args.release_duration,
                        release_position_gain=args.release_gain,
                    ),
                    indent=2,
                )
            )
        elif args.command == "animate-cutting":
            report = json.loads((args.source / "summary.json").read_text())
            if "peripy_commit" in report:
                from .cutting_render import render_cutting

                render_cutting(args.source)
            else:
                from .advanced_render import render_banana_cutting

                render_banana_cutting(args.source)
            print(args.source / "banana_cutting.gif")
        elif args.command == "banana-cutting-disect":
            from .backends.banana_cutting import run_banana_cutting

            print(
                json.dumps(
                    run_banana_cutting(
                        args.output,
                        source=args.backend,
                        dt=args.dt,
                        coupling_dt=args.coupling_dt,
                        duration_s=args.duration,
                        softening_rate=args.softening_rate,
                        fps=args.fps,
                        render=not args.no_render,
                    ),
                    indent=2,
                )
            )
        elif args.command == "banana-cutting":
            from .backends.peridynamic_cutting import run_cutting

            result = run_cutting(
                args.output,
                source=args.backend,
                spacing=args.spacing,
                dt=args.dt,
                coupling_dt=args.coupling_dt,
                duration_s=args.duration,
                critical_stretch=args.critical_stretch,
                knife_cutting=not args.no_knife_severing,
                cut_positions=args.cut_x,
                fps=args.fps,
            )
            if not args.no_render:
                from .cutting_render import render_cutting

                render_cutting(args.output, fps=args.fps)
            print(json.dumps(result, indent=2))
        elif args.command == "robot-demo":
            from .backends.robot import run_robot

            print(
                json.dumps(
                    run_robot(
                        args.output,
                        friction=args.friction,
                        finger_force_n=args.finger_force_n,
                        render=not args.no_render,
                        fps=args.fps,
                    ),
                    indent=2,
                )
            )
        else:
            from .backends.jax_mpm import run_grasp

            print(
                json.dumps(
                    run_grasp(
                        args.output,
                        grid=args.grid,
                        frames=args.frames,
                        friction=args.friction,
                        shock_peak=args.shock_peak,
                        dt_factor=args.dt_factor,
                        young_pa=args.young_pa,
                        gif=args.gif,
                    ),
                    indent=2,
                )
            )
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        parser.exit(2, f"softgriplab: {error}\n")
