"""Geometry-resolved mu/W identification using exact dimensional reduction.

The offline library resolves only eta=W/(mu*L). The shear-modulus scale is
eliminated analytically in the force likelihood; a bounded scalar profile
search recovers eta. No truth parameters are inserted into the library.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import elementwise, least_squares, minimize_scalar

from .complex_contact import HyperelasticContact, LENGTH_M, POISSON, PROBE_SITES

MU_BOUNDS = (4000., 16000.)
W_BOUNDS = (0.4, 6.)
FORCE_NOISE = 0.004
SURFACE_NOISE = 0.000025


def model_signature(model):
    """Cache identity includes solver source, reference mesh and fixed protocol."""
    source = (Path(__file__).parent/"complex_contact.py").read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(source)
    # Ignore platform libm roundoff and byte order, not meaningful mesh changes.
    points = np.round(model.points, decimals=12).astype("<f8")
    points[points == 0] = 0.0
    digest.update(points.tobytes())
    digest.update(np.asarray(model.tets, dtype="<i4").tobytes())
    digest.update(json.dumps([asdict(p) for p in PROBE_SITES], sort_keys=True).encode())
    digest.update(b"scikit-fem-12.0.2")
    return digest.hexdigest()


def features(model, trajectory):
    return trajectory["reaction"][:, 2], model.surface_observations(trajectory["position"])


class ShapeLibrary:
    def __init__(self, ratios, force, surface):
        self.ratios = np.asarray(ratios)
        self.force = PchipInterpolator(np.log(ratios), force, axis=0, extrapolate=False)
        self.surface = PchipInterpolator(np.log(ratios), surface, axis=0, extrapolate=False)

    def predict(self, mu, work, sites):
        eta = work/(mu*LENGTH_M)
        if not self.ratios[0]*(1-1e-12) <= eta <= self.ratios[-1]*(1+1e-12):
            raise ValueError("parameter ratio lies outside the FEM library")
        log_eta = np.log(np.clip(eta, self.ratios[0], self.ratios[-1]))
        f = np.asarray(self.force(log_eta))[sites]*mu*LENGTH_M**2
        u = np.asarray(self.surface(log_eta))[sites]*LENGTH_M
        return f, u

    def fit(self, force, surface, sites, *, use_surface=True):
        """Variable projection with batched SciPy minimization of local basins.

        Scan every interpolation interval to bracket distinct minima, retain
        all grid/boundary candidates, then refine the brackets together. This
        is a multibasin numerical search, not a proof of a global minimum.
        """
        calls = 0
        def evaluate(log_eta, detailed=False):
            nonlocal calls
            shape = np.shape(log_eta)
            x = np.asarray(log_eta).reshape(-1)
            calls += len(x)
            eta = np.exp(x)
            f = np.asarray(self.force(x))[:, sites]*LENGTH_M**2
            lower = np.maximum(MU_BOUNDS[0], W_BOUNDS[0]/(eta*LENGTH_M))
            upper = np.minimum(MU_BOUNDS[1], W_BOUNDS[1]/(eta*LENGTH_M))
            mu = np.clip(np.sum(f*force, axis=(1,2))/np.maximum(np.sum(f*f, axis=(1,2)), 1e-30), lower, upper)
            cost = np.sum(((mu[:,None,None]*f-force)/FORCE_NOISE)**2, axis=(1,2))
            if use_surface:
                u = np.asarray(self.surface(x))[:,sites]*LENGTH_M
                cost += np.sum(((u-surface)/SURFACE_NOISE)**2, axis=(1,2,3,4))
            cost[lower > upper] = np.inf
            if detailed:
                return cost.reshape(shape), mu.reshape(shape), (eta*mu*LENGTH_M).reshape(shape)
            return cost.reshape(shape)
        grid = np.unique(np.concatenate([np.linspace(lo, hi, 33) for lo,hi in
                                         zip(np.log(self.ratios[:-1]), np.log(self.ratios[1:]))]))
        cost = evaluate(grid)
        indices = np.flatnonzero((cost[1:-1] <= cost[:-2]) & (cost[1:-1] <= cost[2:]) &
                                 ((cost[1:-1] < cost[:-2]) | (cost[1:-1] < cost[2:])))+1
        candidates = [(float(np.min(cost)), float(grid[np.argmin(cost)]))]
        if len(indices):
            brackets = (grid[indices-1], grid[indices], grid[indices+1])
            result = elementwise.find_minimum(evaluate, brackets,
                                               tolerances={"xatol": 1e-8, "xrtol": 1e-10}, maxiter=80)
            candidates.extend((float(f), float(x)) for f,x in zip(result.f_x[result.success], result.x[result.success]))
            for i in np.flatnonzero(~result.success):
                fallback = minimize_scalar(lambda x: float(evaluate(x)), bounds=(brackets[0][i], brackets[2][i]),
                                           method="bounded", options={"xatol": 1e-8})
                candidates.append((float(fallback.fun), float(fallback.x)))
        _, best = min(candidates)
        score, mu, work = evaluate(best, True)
        return {"mu_pa": float(mu), "work_j_m2": float(work), "weighted_squared_error": float(score),
                "forward_evaluations": calls}

    def fit_scalar(self, force, surface, sites, *, use_surface=True):
        """Global interval scan + scalar refinement; mu solved exactly at eta."""
        calls = 0
        def evaluate(log_eta, detailed=False):
            nonlocal calls
            calls += 1
            eta = np.exp(log_eta)
            f = np.asarray(self.force(log_eta))[sites]*LENGTH_M**2
            lower = max(MU_BOUNDS[0], W_BOUNDS[0]/(eta*LENGTH_M))
            upper = min(MU_BOUNDS[1], W_BOUNDS[1]/(eta*LENGTH_M))
            if lower > upper:
                return (np.inf, np.nan, np.nan) if detailed else np.inf
            mu = float(np.clip(np.sum(f*force)/max(np.sum(f*f), 1e-30), lower, upper))
            cost = float(np.sum(((mu*f-force)/FORCE_NOISE)**2))
            if use_surface:
                u = np.asarray(self.surface(log_eta))[sites]*LENGTH_M
                cost += float(np.sum(((u-surface)/SURFACE_NOISE)**2))
            return (cost, mu, eta*mu*LENGTH_M) if detailed else cost
        candidates = []
        for lo, hi in zip(np.log(self.ratios[:-1]), np.log(self.ratios[1:])):
            fit = minimize_scalar(evaluate, bounds=(lo, hi), method="bounded", options={"xatol": 2e-5})
            candidates.append((fit.fun, fit.x))
        candidates.extend((evaluate(x), x) for x in np.log(self.ratios[[0, -1]]))
        _, best = min(candidates)
        cost, mu, work = evaluate(best, True)
        return {"mu_pa": mu, "work_j_m2": work, "weighted_squared_error": cost,
                "forward_evaluations": calls}

    def fit_joint(self, force, surface, sites, *, use_surface=True):
        """Strong comparison: bounded, four-start continuous two-parameter fit.

        This uses exactly the same surrogate, observations and parameter bounds.
        """
        lower = np.log([*MU_BOUNDS[:1], *W_BOUNDS[:1]])
        upper = np.log([MU_BOUNDS[1], W_BOUNDS[1]])
        calls = 0
        def residual(x):
            nonlocal calls
            calls += 1
            f, u = self.predict(*np.exp(x), sites)
            result = ((f-force)/FORCE_NOISE).ravel()
            return np.r_[result, ((u-surface)/SURFACE_NOISE).ravel()] if use_surface else result
        candidates = [least_squares(residual, lower+(upper-lower)*s, bounds=(lower, upper),
                                    diff_step=1e-4, max_nfev=100, ftol=1e-8, xtol=1e-8, gtol=1e-8)
                      for s in (0.2, 0.4, 0.6, 0.8)]
        candidates = [x for x in candidates if x.success]
        if not candidates:
            raise RuntimeError("joint continuous fit did not converge")
        best = min(candidates, key=lambda x: np.sum(x.fun*x.fun))
        mu, work = np.exp(best.x)
        return {"mu_pa": float(mu), "work_j_m2": float(work),
                "weighted_squared_error": float(np.sum(best.fun**2)), "forward_evaluations": calls}

    def fit_discrete(self, force, sites):
        candidates = []
        for eta in self.ratios:
            basis = np.asarray(self.force(np.log(eta)))[sites]*LENGTH_M**2
            for mu in np.geomspace(*MU_BOUNDS, 33):
                work = eta*mu*LENGTH_M
                if W_BOUNDS[0] <= work <= W_BOUNDS[1]:
                    cost = float(np.sum(((basis*mu-force)/FORCE_NOISE)**2))
                    candidates.append((cost, mu, work))
        cost, mu, work = min(candidates)
        return {"mu_pa": float(mu), "work_j_m2": float(work),
                "weighted_squared_error": cost, "forward_evaluations": len(candidates)}


def build_shape_library(output, *, cells=(12, 4, 4), count=13):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = HyperelasticContact(cells)
    signature = model_signature(model)
    ratios = np.geomspace(W_BOUNDS[0]/(MU_BOUNDS[1]*LENGTH_M), W_BOUNDS[1]/(MU_BOUNDS[0]*LENGTH_M), count)
    force, surface, timings = [], [], []
    for eta in ratios:
        row_f, row_u = [], []
        started = time.perf_counter()
        for index, site in enumerate(PROBE_SITES):
            cache = output/f"eta_{eta:.9f}_site_{index}.npz"
            cached = False
            if cache.exists():
                with np.load(cache) as data:
                    cached = "signature" in data and str(data["signature"]) == signature
                    if cached:
                        f, u = data["force"], data["surface"]
            if not cached:
                result = model.cycle(eta, site)
                f, u = features(model, result)
                np.savez_compressed(cache, force=f, surface=u, signature=signature,
                                    min_j=min(r["min_j"] for r in result["reports"]),
                                    max_residual=max(r["residual"] for r in result["reports"]))
            row_f.append(f)
            row_u.append(u)
        force.append(row_f)
        surface.append(row_u)
        timings.append(time.perf_counter()-started)
        print(f"  3D library eta={eta:.5f}: {timings[-1]:.1f} s", flush=True)
    np.savez_compressed(output/"library.npz", ratios=ratios, force=force, surface=surface, cells=cells, signature=signature,
                        solve_seconds=timings)
    return model, ShapeLibrary(ratios, np.asarray(force), np.asarray(surface))


def run_shape_identification(output, *, library_count=13, trials=24, seed=41, render=True,
                             library_path=None, validation_materials=6, dense=True):
    output = Path(output)
    if type(validation_materials) is not int or validation_materials < 2 or type(trials) is not int or trials < 1:
        raise ValueError("at least two validation materials and one noise trial are required")
    if type(library_count) is not int or library_count < 5:
        raise ValueError("at least five library ratios are required")
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if library_path is None:
        model, library = build_shape_library(output/"library", count=library_count)
        library_path = output/"library/library.npz"
    else:
        with np.load(library_path, allow_pickle=False) as data:
            model = HyperelasticContact(tuple(data["cells"].astype(int)))
            if "signature" not in data or str(data["signature"]) != model_signature(model):
                raise ValueError("library model metadata is missing or stale; rebuild the library")
            library = ShapeLibrary(data["ratios"], data["force"], data["surface"])
    # Predeclared off-grid materials; evaluations use direct FEM, never the
    # interpolation which performs inference. Repeated noise trials are paired.
    truth_rng = np.random.default_rng(271828)
    truths = np.exp(truth_rng.uniform(np.log([5000., .7]), np.log([15000., 4.8]), (validation_materials, 2)))
    truth_force, truth_surface = [], []
    for index, (mu, work) in enumerate(truths):
        force, surface = [], []
        for site_id, site in enumerate(PROBE_SITES):
            truth_file = output/f"truth_{index}_site_{site_id}.npz"
            if truth_file.exists():
                with np.load(truth_file, allow_pickle=False) as data:
                    if not np.array_equal(data["parameters"], [mu, work]):
                        raise ValueError("cached truth material does not match this validation configuration")
                    if "signature" not in data or str(data["signature"]) != model_signature(model):
                        raise ValueError("cached truth solver/mesh is stale; choose a fresh output directory")
                    result = {"position": data["position"], "reaction": data["reaction"], "centers": data["centers"]}
            else:
                result = model.cycle(work/(mu*LENGTH_M), site)
                np.savez_compressed(truth_file, parameters=[mu, work], signature=model_signature(model), position=result["position"],
                                    reaction=result["reaction"], centers=result["centers"])
            f, u = features(model, result)
            force.append(f*mu*LENGTH_M**2)
            surface.append(u*LENGTH_M)
            if index == 1 and site_id == 1:
                np.savez_compressed(output/"trajectory.npz", position_m=result["position"]*LENGTH_M,
                                    reference_m=model.points*LENGTH_M, tetrahedra=model.tets,
                                    surface_triangles=model.faces, time_s=np.linspace(0, 3, len(f)),
                                    sphere_center_m=result["centers"]*LENGTH_M,
                                    sphere_radius_m=site.radius*LENGTH_M, force_n=f*mu*LENGTH_M**2)
        truth_force.append(force)
        truth_surface.append(surface)
        print(f"  direct off-grid material {index+1}/{len(truths)}", flush=True)
    truth_force, truth_surface = np.asarray(truth_force), np.asarray(truth_surface)
    rng = np.random.default_rng(seed)
    observations_f = truth_force[:, None]+rng.normal(size=(len(truths), trials, *truth_force.shape[1:]))*FORCE_NOISE
    observations_u = truth_surface[:, None]+rng.normal(size=(len(truths), trials, *truth_surface.shape[1:]))*SURFACE_NOISE
    methods = ("profile_force_only", "joint_force_only", "discrete_force_only", "scalar_profile_force_only",
               "profile_force_surface", "joint_force_surface")
    records = []
    for material in range(len(truths)):
        for trial in range(trials):
            f, u = observations_f[material, trial], observations_u[material, trial]
            for method in methods:
                before = time.perf_counter()
                if method == "discrete_force_only":
                    fit = library.fit_discrete(f, [0, 1, 2])
                elif method == "scalar_profile_force_only":
                    fit = library.fit_scalar(f, u, [0,1,2], use_surface=False)
                elif method.startswith("joint"):
                    fit = library.fit_joint(f, u, [0, 1, 2], use_surface=method.endswith("surface"))
                else:
                    fit = library.fit(f, u, [0, 1, 2], use_surface=method.endswith("surface"))
                records.append({"material": material, "noise_trial": trial, "method": method,
                                "seconds": time.perf_counter()-before, **fit,
                                "relative_mu_error": abs(fit["mu_pa"]/truths[material, 0]-1),
                                "relative_work_error": abs(fit["work_j_m2"]/truths[material, 1]-1)})
    summary = {}
    for method in methods:
        rows = [r for r in records if r["method"] == method]
        summary[method] = {"mean_mu_relative_error": float(np.mean([r["relative_mu_error"] for r in rows])),
                           "mean_work_relative_error": float(np.mean([r["relative_work_error"] for r in rows])),
                           "median_online_seconds": float(np.median([r["seconds"] for r in rows])),
                           "median_forward_evaluations": float(np.median([r["forward_evaluations"] for r in rows])),
                           "mean_whitened_fit_rmse": float(np.mean([np.sqrt(r["weighted_squared_error"] /
                               (truth_force[0].size+(truth_surface[0].size if method.endswith("surface") else 0))) for r in rows]))}
    dense_report = None
    if dense:
        dense_model = HyperelasticContact((20, 8, 8))
        dense_offsets = np.r_[np.linspace(.13, -.16, 19), np.linspace(-.16, .24, 25)[1:]]
        # The replay uses the scalar profile ablation; the batched method solves
        # the same likelihood and their parameter/objective agreement is tested.
        fitted = next(r for r in records if r["material"] == 1 and r["noise_trial"] == 0 and r["method"] == "scalar_profile_force_only")
        dense_states, dense_forces, checks = [], [], []
        dense_parameters = np.array([truths[1], [fitted["mu_pa"], fitted["work_j_m2"]]])
        reused = False
        dense_file = output/"dense_validation.npz"
        if dense_file.exists():
            with np.load(dense_file, allow_pickle=False) as data:
                reused = ("signature" in data and str(data["signature"]) == model_signature(dense_model)
                          and "parameters" in data and np.allclose(data["parameters"], dense_parameters, rtol=1e-10, atol=1e-10))
                if reused:
                    dense_states, dense_forces = data["position_m"], data["force_n"]
                    checks = json.loads(str(data["checks_json"]))
        if not reused:
            for name, mu, work in (("truth", *truths[1]), ("identified", fitted["mu_pa"], fitted["work_j_m2"])):
                print(f"  dense FEM validation: {name}", flush=True)
                result = dense_model.cycle(work/(mu*LENGTH_M), PROBE_SITES[1], offsets=dense_offsets)
                dense_states.append(result["position"]*LENGTH_M)
                dense_forces.append(result["reaction"][:, 2]*mu*LENGTH_M**2)
                checks.append({"case": name, "min_det_f": min(r["min_j"] for r in result["reports"]),
                               "maximum_equilibrium_residual": max(r["residual"] for r in result["reports"])})
            np.savez_compressed(dense_file, position_m=dense_states, force_n=dense_forces,
                                reference_m=dense_model.points*LENGTH_M, tetrahedra=dense_model.tets,
                                surface_triangles=dense_model.faces, sphere_center_m=result["centers"]*LENGTH_M,
                                sphere_radius_m=PROBE_SITES[1].radius*LENGTH_M, time_s=np.linspace(0, 3, len(dense_offsets)),
                                signature=model_signature(dense_model), parameters=dense_parameters,
                                checks_json=json.dumps(checks))
        dense_report = {"checks": checks, "nodes": len(dense_model.points), "tetrahedra": len(dense_model.tets),
                        "inference_method": "scalar_profile_force_only",
                        "states_reused": reused,
                        "force_relative_l2_error": float(np.linalg.norm(dense_forces[1]-dense_forces[0])/np.linalg.norm(dense_forces[0])),
                        "mean_absolute_force_error_n": float(np.mean(np.abs(dense_forces[1]-dense_forces[0]))),
                        "pull_off_peak_relative_error": float(abs(np.min(dense_forces[1])/np.min(dense_forces[0])-1)),
                        "position_rmse_m": float(np.sqrt(np.mean((dense_states[1]-dense_states[0])**2)))}
    with np.load(library_path, allow_pickle=False) as library_data:
        library_generation_seconds = float(np.sum(library_data["solve_seconds"]))
    cost_by_method = {method: np.array([r["weighted_squared_error"] for r in records if r["method"] == method])
                      for method in methods}
    performance = {
        "online_speedup_over_multistart_same_force_data": summary["joint_force_only"]["median_online_seconds"]/summary["profile_force_only"]["median_online_seconds"],
        "online_speedup_over_scalar_profile": summary["scalar_profile_force_only"]["median_online_seconds"]/summary["profile_force_only"]["median_online_seconds"],
        "maximum_objective_excess_over_scalar_profile": float(np.max(cost_by_method["profile_force_only"]-cost_by_method["scalar_profile_force_only"])),
        "maximum_objective_excess_over_multistart": float(np.max(cost_by_method["profile_force_only"]-cost_by_method["joint_force_only"])),
        "scope": "Paired online inverse timings using the same precomputed FEM library and observations; offline construction excluded."
    }
    report = {"model": "3D compressible Neo-Hookean + finite-range reversible surface adhesion",
              "backend": "scikit-fem + SciPy sparse Newton with energy line search",
              "poisson_known": POISSON, "length_scale_m": LENGTH_M, "seed": seed,
              "independent_materials": len(truths), "paired_noise_repeats_per_material": trials,
              "truth_parameters": truths.tolist(), "noise": {"force_n": FORCE_NOISE, "surface_m": SURFACE_NOISE},
              "summary": summary, "elapsed_seconds_before_render_this_run": time.perf_counter()-start,
              "library_generation_run_seconds": library_generation_seconds,
              "library_ratio_samples": len(library.ratios), "library_probe_cycles": len(library.ratios)*len(PROBE_SITES),
              "performance": performance,
              "learning_regime": f"No neural network or large training set: {len(library.ratios)} mechanics-ratio samples at three probe sites; noise repeats are validation only.",
              "validation_material_seed": 271828, "dense_validation": dense_report,
              "library_sha256": hashlib.sha256(Path(library_path).read_bytes()).hexdigest(),
              "library_reused": str(library_path),
              "source_sha256": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                for name in ("complex_contact.py", "shape_identification.py")},
              "limitations": ["Known complex geometry, known fixtures and Poisson ratio; only mu and W are inferred.",
                              "Gravity-free quasi-static load path; no dynamics, rate effects, friction or irreversible damage.",
                              "Identification truth uses direct off-grid FEM on the inference mesh. Dense transfer is checked for one material only.",
                              "Lumped reference-surface integration; contact and mesh convergence still require study.",
                              "Noise repeats do not count as independent material or geometry tests.",
                              "Online timing excludes offline library construction and direct FEM validation; a reused library is explicitly named.",
                              "The current benchmark compares inverse methods; it does not establish active-location selection superiority."]}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    import csv
    with (output/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(output/"observations.npz", truth_parameters=truths, force=observations_f,
                        surface=observations_u, clean_force=truth_force, clean_surface=truth_surface)
    if render:
        from .shape_render import render_shape_identification
        render_shape_identification(output)
    return report
