"""Unknown-surface evidence without granting the inverse solver true geometry."""

import json
import hashlib
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("skfem")
from softgriplab.complex_contact import LENGTH_M
from softgriplab.surface_geometry import ScannedContact, TubeSurface, plan_probes, reconstruct_surface, synthetic_surface


def test_unlabelled_partial_scan_reconstructs_a_held_out_surface():
    rng = np.random.default_rng(154)
    truth = synthetic_surface()
    x = rng.uniform(-2.995,2.995,360)
    angles = rng.uniform(-.15*np.pi,1.15*np.pi,len(x))
    observed = truth.surface(x,angles)*LENGTH_M+rng.normal(scale=4e-5,size=(len(x),3))
    observed[:,0] = np.clip(observed[:,0],-3*LENGTH_M,3*LENGTH_M)
    # Only unordered measured xyz enters reconstruction.
    rng.shuffle(observed)
    fitted, report = reconstruct_surface(observed)
    x, angle = np.meshgrid(np.linspace(-2.8,2.8,61),np.linspace(0,2*np.pi,81),indexing="ij")
    error = np.linalg.norm(fitted.surface(x,angle)-truth.surface(x,angle),axis=-1)*LENGTH_M
    assert np.sqrt(np.mean(error**2)) < 1.8e-4
    assert report["scan_normal_rmse_m"] < 7e-5
    # Unobserved lower-arc recovery is tested only within the stated shape prior.
    hidden = (angle > 1.15*np.pi) & (angle < 1.85*np.pi)
    assert np.sqrt(np.mean(error[hidden]**2)) < 2.5e-4


def test_reconstructed_reference_is_stress_free_and_balances_contact():
    geometry = synthetic_surface()
    model = ScannedContact(geometry,(8,4,4))
    energy, gradient, _, _, minimum = model.evaluate(np.zeros(model.basis.N),[0,0,10],.65,.02)
    assert abs(energy) < 1e-10 and minimum == pytest.approx(1.,abs=1e-12)
    np.testing.assert_allclose(gradient,0.,atol=1e-11)
    probe = plan_probes(geometry)[1]
    u, reaction, report = model.equilibrium(probe.center(-.04),probe.radius,.015)
    _, residual, _, _, _ = model.evaluate(u,probe.center(-.04),probe.radius,.015)
    fixture_force = -residual[model.basis.nodal_dofs[:,model.fixed_nodes]].sum(axis=1)
    np.testing.assert_allclose(fixture_force+reaction,0.,atol=2e-5)
    assert report["min_j"] > 0


def test_insufficient_axial_coverage_is_rejected():
    rng = np.random.default_rng(2)
    points = np.column_stack((rng.uniform(-.01,.01,100),rng.normal(0,.005,(100,2))))
    with pytest.raises(ValueError,match="axial extent"):
        reconstruct_surface(points)


def test_unknown_surface_archive_recomputes_geometry_and_material_errors():
    archive = Path(__file__).resolve().parents[1]/"docs/results/unknown_surface"
    if not (archive/"report.json").exists():
        pytest.skip("unknown-surface benchmark not archived")
    import csv
    report = json.loads((archive/"report.json").read_text())
    for name,digest in json.loads((archive/"sha256.json").read_text()).items():
        assert hashlib.sha256((archive/name).read_bytes()).hexdigest() == digest
    with np.load(archive/"scan.npz",allow_pickle=False) as data:
        geometry = TubeSurface(data["stations"],data["values"])
        residual = geometry.residual(data["points_m"]/LENGTH_M)*LENGTH_M
        assert np.sqrt(np.mean(residual**2)) == pytest.approx(report["surface"]["scan_normal_rmse_m"])
        error = np.linalg.norm(data["evaluation_reconstruction_m"]-data["evaluation_truth_m"],axis=-1)
        assert np.sqrt(np.mean(error**2)) == pytest.approx(report["surface"]["surface_rmse_m"])
        assert np.sqrt(np.mean(error[data["hidden_mask"]]**2)) == pytest.approx(report["surface"]["hidden_surface_rmse_m"])
    with np.load(archive/"observations.npz",allow_pickle=False) as data:
        truth = data["materials"]
        origins = data["probe_origins"]
    np.testing.assert_allclose(origins,np.array([p.origin for p in plan_probes(geometry)]),atol=1e-12)
    with (archive/"metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        mu, work = truth[int(row["material"])]
        assert float(row["mu_relative_error"]) == pytest.approx(abs(float(row["mu_pa"])/mu-1))
        assert float(row["work_relative_error"]) == pytest.approx(abs(float(row["work_j_m2"])/work-1))
    for method,summary in report["summary"].items():
        group = [r for r in rows if r["method"]==method]
        assert len(group) == len(truth)*report["noise_trials"]
        assert np.mean([float(r["mu_relative_error"]) for r in group]) == pytest.approx(summary["mean_mu_relative_error"])
        assert np.mean([float(r["work_relative_error"]) for r in group]) == pytest.approx(summary["mean_work_relative_error"])


def test_unknown_surface_dense_frames_are_equilibria_on_each_reference_shape():
    archive = Path(__file__).resolve().parents[1]/"docs/results/unknown_surface"
    if not (archive/"dense_validation.npz").exists():
        pytest.skip("dense unknown-surface archive unavailable")
    with np.load(archive/"scan.npz",allow_pickle=False) as scan:
        recovered = TubeSurface(scan["stations"],scan["values"])
    with np.load(archive/"dense_validation.npz",allow_pickle=False) as data:
        for case,geometry in enumerate((synthetic_surface(),recovered)):
            model = ScannedContact(geometry,(20,8,8))
            mu, work = data["parameters"][case]
            for frame in (18,37):
                u = np.zeros(model.basis.N)
                u[model.basis.nodal_dofs] = (data["position_m"][case,frame]/LENGTH_M-model.points).T
                _, residual, _, reaction, j = model.evaluate(u,data["sphere_center_m"][frame]/LENGTH_M,
                    float(data["sphere_radius_m"])/LENGTH_M,work/(mu*LENGTH_M),tangent=False)
                assert j > 0 and np.max(np.abs(residual[model.free])) < 2.01e-6
                assert reaction[2]*mu*LENGTH_M**2 == pytest.approx(data["force_n"][case,frame],abs=1e-8)


def test_failed_large_increment_is_replaced_by_solved_continuation_states():
    archive = Path(__file__).resolve().parents[1]/"docs/results/unknown_surface"
    if not (archive/"adaptive_cycle.npz").exists():
        pytest.skip("continuation audit archive unavailable")
    with np.load(archive/"scan.npz",allow_pickle=False) as scan:
        geometry = TubeSurface(scan["stations"],scan["values"])
    model = ScannedContact(geometry,(12,4,4))
    with np.load(archive/"adaptive_cycle.npz",allow_pickle=False) as data:
        assert str(data["protocol"]) == "bisection-v1"
        assert int(data["subdivisions"]) > 0
        for frame,nodes in enumerate(data["position"]):
            u = np.zeros(model.basis.N)
            u[model.basis.nodal_dofs] = (nodes-model.points).T
            _, residual, _, reaction, j = model.evaluate(u,data["centers"][frame],.65,float(data["eta"]),tangent=False)
            assert j > 0 and np.max(np.abs(residual[model.free])) < 2.01e-6
            np.testing.assert_allclose(reaction,data["reaction"][frame],atol=1e-9)
