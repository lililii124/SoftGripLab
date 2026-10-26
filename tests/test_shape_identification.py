"""Audit inverse-method evidence and the archived 3D equilibrium replay."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("skfem")
from softgriplab.complex_contact import HyperelasticContact, LENGTH_M
from softgriplab.shape_identification import FORCE_NOISE, ShapeLibrary, model_signature

ARCHIVE = Path(__file__).resolve().parents[1]/"docs/results/shape_identification"


@pytest.fixture(scope="module")
def archive():
    if not (ARCHIVE/"report.json").exists():
        pytest.skip("3D identification archive unavailable")
    report = json.loads((ARCHIVE/"report.json").read_text())
    with np.load(ARCHIVE/"library.npz", allow_pickle=False) as data:
        library = ShapeLibrary(data["ratios"], data["force"], data["surface"])
        assert str(data["signature"]) == model_signature(HyperelasticContact(tuple(data["cells"])))
    with (ARCHIVE/"metrics.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    return report, library, records


def test_archived_errors_are_recomputed_from_independent_truth(archive):
    report, library, rows = archive
    assert hashlib.sha256((ARCHIVE/"library.npz").read_bytes()).hexdigest() == report["library_sha256"]
    truth = np.asarray(report["truth_parameters"])
    ratios = truth[:, 1]/(truth[:, 0]*LENGTH_M)
    assert all(np.min(np.abs(library.ratios-eta)) > 1e-10 for eta in ratios)
    for row in rows:
        mu, work = truth[int(row["material"])]
        assert float(row["relative_mu_error"]) == pytest.approx(abs(float(row["mu_pa"])/mu-1), abs=1e-12)
        assert float(row["relative_work_error"]) == pytest.approx(abs(float(row["work_j_m2"])/work-1), abs=1e-12)
    for method, summary in report["summary"].items():
        group = [r for r in rows if r["method"] == method]
        assert len(group) == report["independent_materials"]*report["paired_noise_repeats_per_material"]
        assert np.mean([float(r["relative_mu_error"]) for r in group]) == pytest.approx(summary["mean_mu_relative_error"])
        assert np.mean([float(r["relative_work_error"]) for r in group]) == pytest.approx(summary["mean_work_relative_error"])


def test_profile_fit_reproduces_saved_estimate_and_strong_baseline_cost(archive):
    report, library, rows = archive
    with np.load(ARCHIVE/"observations.npz", allow_pickle=False) as data:
        f, u = data["force"][1, 0], data["surface"][1, 0]
    fitted = library.fit(f, u, [0,1,2], use_surface=False)
    row = next(r for r in rows if r["material"]=="1" and r["noise_trial"]=="0" and r["method"]=="profile_force_only")
    assert fitted["mu_pa"] == pytest.approx(float(row["mu_pa"]), rel=1e-8)
    assert fitted["work_j_m2"] == pytest.approx(float(row["work_j_m2"]), rel=1e-8)
    baseline = library.fit_joint(f, u, [0,1,2], use_surface=False)
    assert fitted["weighted_squared_error"] <= baseline["weighted_squared_error"]+0.01
    # Exact dimensional similarity of the response model, including all observed
    # spatial displacement components; this is not a continuum-convergence claim.
    f1, u1 = library.predict(6000., 1.8, [0,1,2])
    f2, u2 = library.predict(12000., 3.6, [0,1,2])
    np.testing.assert_allclose(f2, 2*f1, atol=1e-14)
    np.testing.assert_allclose(u2, u1, atol=1e-14)


def test_all_archived_force_fits_preserve_the_same_objective(archive):
    report, library, rows = archive
    methods = ("profile_force_only", "scalar_profile_force_only", "joint_force_only")
    with np.load(ARCHIVE/"observations.npz", allow_pickle=False) as data:
        observed = data["force"]
    costs = {method: {} for method in methods}
    for row in rows:
        if row["method"] not in methods:
            continue
        key = int(row["material"]), int(row["noise_trial"])
        predicted, _ = library.predict(float(row["mu_pa"]), float(row["work_j_m2"]), [0,1,2])
        score = float(np.sum(((predicted-observed[key])/FORCE_NOISE)**2))
        assert score == pytest.approx(float(row["weighted_squared_error"]), abs=1e-7)
        costs[row["method"]][key] = score
    for baseline, field in (("scalar_profile_force_only", "maximum_objective_excess_over_scalar_profile"),
                            ("joint_force_only", "maximum_objective_excess_over_multistart")):
        assert costs[baseline].keys() == costs["profile_force_only"].keys()
        difference = max(score-costs[baseline][key] for key,score in costs["profile_force_only"].items())
        assert difference <= 1e-6
        assert difference == pytest.approx(report["performance"][field], abs=1e-7)
    # Audit the reported arithmetic without asserting machine-dependent speed.
    summary = report["summary"]
    ratio = summary["joint_force_only"]["median_online_seconds"]/summary["profile_force_only"]["median_online_seconds"]
    assert ratio == pytest.approx(report["performance"]["online_speedup_over_multistart_same_force_data"])


def test_dense_replay_force_is_derived_from_saved_equilibrium(archive):
    report, _, _ = archive
    model = HyperelasticContact((20,8,8))
    with np.load(ARCHIVE/"dense_validation.npz", allow_pickle=False) as data:
        assert str(data["signature"]) == model_signature(model)
        np.testing.assert_allclose(data["reference_m"], model.points*LENGTH_M)
        for case in (0,1):
            mu, work = data["parameters"][case]
            for frame in (18,37):
                u = np.zeros(model.basis.N)
                u[model.basis.nodal_dofs] = (data["position_m"][case,frame]/LENGTH_M-model.points).T
                _, residual, _, reaction, min_j = model.evaluate(u, data["sphere_center_m"][frame]/LENGTH_M,
                                                               float(data["sphere_radius_m"])/LENGTH_M,
                                                               work/(mu*LENGTH_M), tangent=False)
                assert min_j > 0
                assert np.max(np.abs(residual[model.free])) < 2.01e-6
                assert reaction[2]*mu*LENGTH_M**2 == pytest.approx(data["force_n"][case,frame], abs=1e-8)
        relative = np.linalg.norm(data["force_n"][1]-data["force_n"][0])/np.linalg.norm(data["force_n"][0])
        assert relative == pytest.approx(report["dense_validation"]["force_relative_l2_error"])
    render = json.loads((ARCHIVE/"render.json").read_text())
    assert hashlib.sha256((ARCHIVE/render["trajectory"]).read_bytes()).hexdigest() == render["trajectory_sha256"]
