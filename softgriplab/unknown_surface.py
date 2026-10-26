"""Point-cloud reconstruction followed by force-based material identification.

This is a conditional, two-stage baseline, not joint shape/material inference.
Every method sees the same prescribed probe poses and force observations.
"""

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .complex_contact import HyperelasticContact, LENGTH_M
from .shape_identification import FORCE_NOISE, MU_BOUNDS, W_BOUNDS, ShapeLibrary, model_signature
from .surface_geometry import ScannedContact, plan_probes, reconstruct_surface, synthetic_surface


def adaptive_cycle(model, eta, probe, offsets=None):
    """Retry failed load increments by bisection; never interpolate deformation."""
    if offsets is None:
        offsets = np.r_[np.linspace(.13,-.16,12),np.linspace(-.16,.24,16)[1:]]
    origin = probe.center(0)
    centers = origin+np.asarray(offsets)[:,None]*np.array([0.,0.,1.])
    reports, positions, reactions = [], [], []
    subdivisions = 0
    def advance(u, previous, target, depth=0):
        nonlocal subdivisions
        try:
            state, force, report = model.equilibrium(target, probe.radius, eta, u)
            reports.append(report)
            return state, force
        except RuntimeError as error:
            if depth >= 8:
                raise RuntimeError("adaptive loading failed after eight bisections") from error
            subdivisions += 1
            middle = (previous+target)/2
            midpoint, _ = advance(u,previous,middle,depth+1)
            return advance(midpoint,middle,target,depth+1)
    u = np.zeros(model.basis.N)
    previous = origin+[0,0,.4]
    for center in centers:
        u, reaction = advance(u,previous,center)
        positions.append(model.points+u[model.basis.nodal_dofs].T)
        reactions.append(reaction)
        previous = center
    return {"position":np.asarray(positions),"reaction":np.asarray(reactions),"centers":centers,
            "min_j":min(r["min_j"] for r in reports),
            "max_residual":max(r["residual"] for r in reports),"subdivisions":subdivisions}


def cache_signature(model, probes, offsets=None):
    digest = hashlib.sha256(model_signature(model).encode())
    digest.update((Path(__file__).parent/"surface_geometry.py").read_bytes().replace(b"\r\n", b"\n"))
    digest.update(json.dumps([asdict(p) for p in probes], sort_keys=True).encode())
    digest.update(json.dumps(None if offsets is None else np.asarray(offsets).tolist()).encode())
    return digest.hexdigest()


def cached_cycle(path, model, probe, eta, signature, offsets=None):
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if str(data["signature"]) != signature or float(data["eta"]) != eta:
                raise ValueError(f"stale cycle cache: {path}; use a fresh output directory")
            if "protocol" in data and str(data["protocol"]) not in ("direct", "bisection-v1"):
                raise ValueError("unsupported cached loading protocol")
            record = {key: data[key].copy() for key in ("position", "reaction", "centers", "min_j", "max_residual")}
            record["subdivisions"] = int(data["subdivisions"]) if "subdivisions" in data else 0
            return record
    protocol = "direct"
    try:
        result = model.cycle(eta, probe, offsets=offsets)
        record = {key: result[key] for key in ("position", "reaction", "centers")}
        record.update(min_j=min(r["min_j"] for r in result["reports"]),
                      max_residual=max(r["residual"] for r in result["reports"]),subdivisions=0)
    except RuntimeError:
        print(f"  adaptive load subdivision: {path.name}",flush=True)
        record = adaptive_cycle(model,eta,probe,offsets)
        protocol = "bisection-v1"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, signature=signature, eta=eta, protocol=protocol, **record)
    return record


def response_library(output, model, probes, count):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = cache_signature(model, probes)
    ratios = np.geomspace(W_BOUNDS[0]/(MU_BOUNDS[1]*LENGTH_M), W_BOUNDS[1]/(MU_BOUNDS[0]*LENGTH_M), count)
    forces, checks = [], []
    for index, eta in enumerate(ratios):
        row = []
        for site, probe in enumerate(probes):
            result = cached_cycle(output/f"ratio_{index}_site_{site}.npz", model, probe, float(eta), signature)
            row.append(result["reaction"][:,2])
            checks.append([float(result["min_j"]), float(result["max_residual"]),int(result["subdivisions"])])
        forces.append(row)
        print(f"  {output.name}: FEM ratio {index+1}/{count}", flush=True)
    force = np.asarray(forces)
    # This benchmark observes force only; no synthetic surface-displacement
    # channels are passed into the inverse solver.
    surface_placeholder = np.zeros((*force.shape, 1, 3))
    np.savez_compressed(output/"library.npz", ratios=ratios, force=force, surface=surface_placeholder,
                        signature=signature, reference=model.points, checks=checks)
    return ShapeLibrary(ratios, force, surface_placeholder)


def run_unknown_surface(output, *, scan_path=None, library_count=13, trials=12, seed=73,
                        dense=True, render=True):
    """Synthetic controlled benchmark. External scans are reconstructed separately.

    True geometry is private to the observation generator and evaluation. It is
    not available to the surface fit, tool planner or material inverse method.
    """
    if scan_path is not None:
        raise ValueError("benchmark truth is synthetic; use reconstruct_surface for external xyz scans")
    if library_count < 5 or trials < 1:
        raise ValueError("library_count >= 5 and trials >= 1 are required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    truth_surface = synthetic_surface()
    x = rng.uniform(-2.995, 2.995, 360)
    angles = rng.uniform(-.15*np.pi, 1.15*np.pi, len(x))
    scan = truth_surface.surface(x, angles)*LENGTH_M
    scan += rng.normal(scale=4e-5, size=scan.shape)
    scan[:,0] = np.clip(scan[:,0], -3*LENGTH_M, 3*LENGTH_M)
    reconstructed, scan_report = reconstruct_surface(scan)
    # The unseen lower arc and the measured arc are evaluated separately.
    tx, ta = np.meshgrid(np.linspace(-2.95, 2.95, 101), np.linspace(0,2*np.pi,121), indexing="ij")
    true_surface_points = truth_surface.surface(tx,ta)
    reconstructed_points = reconstructed.surface(tx,ta)
    error = np.linalg.norm(reconstructed_points-true_surface_points, axis=-1)*LENGTH_M
    hidden = (ta > 1.15*np.pi) & (ta < 1.85*np.pi)
    scan_report.update(surface_rmse_m=float(np.sqrt(np.mean(error**2))),
                       hidden_surface_rmse_m=float(np.sqrt(np.mean(error[hidden]**2))),
                       evaluation="same axial/angle coordinates, independent dense validation surface; not used by fitter")
    probes = plan_probes(reconstructed)
    np.savez_compressed(output/"scan.npz", points_m=scan, stations=reconstructed.stations,
                        values=reconstructed.values, evaluation_truth_m=true_surface_points*LENGTH_M,
                        evaluation_reconstruction_m=reconstructed_points*LENGTH_M,
                        hidden_mask=hidden)
    print(f"  point cloud reconstructed; surface RMSE {scan_report['surface_rmse_m']*1e3:.3f} mm", flush=True)
    nominal = HyperelasticContact((12,4,4))
    recovered = ScannedContact(reconstructed, (12,4,4))
    true_model = ScannedContact(truth_surface, (12,4,4))
    libraries = {}
    library_times = {}
    for name, model in (("nominal_geometry", nominal), ("reconstructed_geometry", recovered)):
        before = time.perf_counter()
        libraries[name] = response_library(output/name, model, probes, library_count)
        library_times[name] = time.perf_counter()-before
    materials = np.exp(np.random.default_rng(90210).uniform(np.log([6000., .8]), np.log([14000., 3.6]), (3,2)))
    clean = []
    true_signature = cache_signature(true_model, probes)
    for material, (mu, work) in enumerate(materials):
        row = []
        for site, probe in enumerate(probes):
            result = cached_cycle(output/f"truth/material_{material}_site_{site}.npz", true_model, probe,
                                  float(work/(mu*LENGTH_M)), true_signature)
            row.append(result["reaction"][:,2]*mu*LENGTH_M**2)
        clean.append(row)
        print(f"  independent material {material+1}/{len(materials)}", flush=True)
    clean = np.asarray(clean)
    observations = clean[:,None]+rng.normal(scale=FORCE_NOISE, size=(len(materials),trials,*clean.shape[1:]))
    records = []
    for name, library in libraries.items():
        for material, (mu, work) in enumerate(materials):
            for trial in range(trials):
                fit = library.fit(observations[material,trial], None, [0,1,2], use_surface=False)
                records.append({"method": name, "material": material, "trial": trial, **fit,
                                "mu_relative_error": abs(fit["mu_pa"]/mu-1),
                                "work_relative_error": abs(fit["work_j_m2"]/work-1)})
    summary = {}
    for name in libraries:
        group = [r for r in records if r["method"] == name]
        summary[name] = {"mean_mu_relative_error": float(np.mean([r["mu_relative_error"] for r in group])),
                         "mean_work_relative_error": float(np.mean([r["work_relative_error"] for r in group])),
                         "mean_whitened_force_rmse": float(np.mean([np.sqrt(r["weighted_squared_error"]/clean[0].size) for r in group]))}
    np.savez_compressed(output/"observations.npz", materials=materials, force=observations, clean_force=clean,
                        probe_origins=[p.origin for p in probes], probe_radius=probes[0].radius)
    with (output/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {"scope": "two-stage geometry then material inversion; one synthetic geometry, three independent materials",
              "surface": scan_report, "summary": summary, "seed": seed, "noise_trials": trials,
              "library_ratios_per_geometry": library_count, "probe_sites": len(probes),
              "library_seconds_this_run_including_cache_reads": library_times,
              "loading": "failed increments use equilibrium-resolved bisection; prescribed observation poses do not change",
              "learning": "no neural training; fixed small surface fit and FEM ratio libraries",
              "dense_validation": None,
              "limitations": ["Known axial registration, end planes, solid topology and swept-ellipse prior.",
                              "Hidden surface completion depends on the geometric prior; arbitrary unseen shapes are not identifiable.",
                              "Reconstruction is frozen during material fitting; geometric uncertainty is not marginalized.",
                              "Homogeneous Neo-Hookean modulus and adhesion work; known Poisson ratio and interface range.",
                              "No real scanner, experiment, robot or arbitrary-shape generalization is validated.",
                              "Nominal-geometry fitting is a mismatch ablation, not a strong joint-geometry optimization baseline.",
                              "Both libraries use the exact same tool poses planned from the reconstructed scan."]}
    if dense:
        offsets = np.r_[np.linspace(.13,-.16,19),np.linspace(-.16,.24,25)[1:]]
        fitted = next(r for r in records if r["method"]=="reconstructed_geometry" and r["material"]==0 and r["trial"]==0)
        parameters = np.array([materials[0], [fitted["mu_pa"], fitted["work_j_m2"]]])
        states, forces, checks = [], [], []
        for case, (geometry, (mu, work)) in enumerate(zip((truth_surface,reconstructed),parameters)):
            print(f"  dense unknown-surface replay {case+1}/2", flush=True)
            model = ScannedContact(geometry, (20,8,8))
            result = cached_cycle(output/f"dense_case_{case}.npz", model, probes[1], float(work/(mu*LENGTH_M)),
                                  cache_signature(model,probes,offsets), offsets)
            states.append(result["position"]*LENGTH_M)
            forces.append(result["reaction"][:,2]*mu*LENGTH_M**2)
            checks.append({"min_j":float(result["min_j"]), "maximum_residual":float(result["max_residual"])})
        np.savez_compressed(output/"dense_validation.npz", position_m=states, force_n=forces, parameters=parameters,
                            tetrahedra=model.tets, surface_triangles=model.faces, sphere_center_m=result["centers"]*LENGTH_M,
                            sphere_radius_m=probes[1].radius*LENGTH_M, time_s=np.linspace(0,3,len(offsets)))
        report["dense_validation"] = {"checks":checks,
            "force_relative_l2_error":float(np.linalg.norm(forces[1]-forces[0])/np.linalg.norm(forces[0])),
            "mean_absolute_force_error_n":float(np.mean(np.abs(forces[1]-forces[0]))),
            "position_rmse_m":float(np.sqrt(np.mean((states[1]-states[0])**2)))}
        if render:
            from .shape_render import render_shape_identification
            render_shape_identification(output)
    report["elapsed_seconds_this_run"] = time.perf_counter()-started
    report["source_sha256"] = {name:hashlib.sha256((Path(__file__).parent/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
                                for name in ("surface_geometry.py","unknown_surface.py","complex_contact.py","shape_identification.py")}
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report
