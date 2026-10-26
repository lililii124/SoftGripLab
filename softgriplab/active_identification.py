"""Finite-ensemble, task-directed Bayesian probing of axial adhesive contact.

All policies receive the same initial observation, action library, noise model
and probe budget. Task labels used for acquisition are MODEL predictions only;
held-out truth labels are used exclusively by the evaluator.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, xlogy
from scipy.stats import qmc

from .active_contact import CALIBRATION, PROBES, ContactSpec, Release, simulate


BOUNDS = np.array([[12000, 65000], [8000, 85000], [0.02, 0.5], [0.8, 3.2]])
NOISE = np.array([0.008, 0.00003])  # independent force [N], marker position [m]
ACCELERATIONS = np.linspace(4, 100, 17)
POLICIES = ("task", "information", "fixed", "random")


def observations(trajectory):
    return np.stack((trajectory["force_n"], trajectory["marker_m"]), axis=-1)


def log_posterior(log_weights, predictions, observed, noise=NOISE):
    residual = (np.asarray(predictions)-np.asarray(observed))/noise
    scores = np.asarray(log_weights) - 0.5*np.sum(residual**2, axis=(-2, -1))
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite likelihood or prior")
    return scores-logsumexp(scores)


def binary_entropy(probability):
    p = np.clip(probability, 0, 1)
    return -xlogy(p, p)-xlogy(1-p, 1-p)


def acquisition_scores(log_weights, predictions, task_labels, *, objective="task",
                       draws=48, seed=0, noise=NOISE):
    """Expected posterior entropy, integrated with common Monte Carlo draws.

    Returned scores are expected entropy REDUCTIONS (larger is better).
    The generic information policy targets the finite hypothesis index.
    """
    if objective not in ("task", "information") or type(draws) is not int or draws < 2:
        raise ValueError("objective must be task/information; draws must be >= 2")
    weights = np.exp(np.asarray(log_weights)-logsumexp(log_weights))
    predictions, labels = np.asarray(predictions), np.asarray(task_labels, float)
    if predictions.ndim != 4 or predictions.shape[1] != len(weights):
        raise ValueError("predictions require action,hypothesis,time,sensor dimensions")
    if labels.ndim != 2 or labels.shape[0] != len(weights):
        raise ValueError("task labels require hypothesis,task dimensions")
    rng = np.random.default_rng(seed)
    latent = rng.choice(len(weights), draws, p=weights)
    epsilon = rng.normal(size=(draws, *predictions.shape[2:]))*noise
    prior_entropy = (float(np.mean(binary_entropy(weights@labels))) if objective == "task"
                     else float(-np.sum(xlogy(weights, weights))))
    scores = []
    for response in predictions:
        hypothetical = response[latent] + epsilon
        # Chunked by hypothetical observation to avoid actions*N*N allocations.
        posterior = np.stack([np.exp(log_posterior(log_weights, response, y, noise))
                              for y in hypothetical])
        entropy = (np.mean(binary_entropy(posterior@labels), axis=1) if objective == "task"
                   else -np.sum(xlogy(posterior, posterior), axis=1))
        scores.append(prior_entropy-float(np.mean(entropy)))
    return np.asarray(scores)


def make_bank(count=512, seed=7):
    if type(count) is not int or count < 32 or count & (count-1):
        raise ValueError("hypotheses must be a power of two >= 32")
    unit = qmc.Sobol(4, scramble=True, seed=seed).random_base2(int(np.log2(count)))
    return np.exp(np.log(BOUNDS[:, 0]) + unit*np.log(BOUNDS[:, 1]/BOUNDS[:, 0]))


def _metrics(log_weights, task_labels, truth):
    weights = np.exp(log_weights)
    probability = weights@task_labels
    ess = float(1/np.sum(weights**2))
    return {"brier": float(np.mean((probability-truth)**2)),
            "accuracy": float(np.mean((probability >= 0.5) == truth)),
            "task_entropy_nats": float(np.mean(binary_entropy(probability))),
            "effective_hypotheses": ess,
            "finite_bank_degenerate": bool(ess < 5)}


def _ambiguous_pair(log_weights, prediction, observed, labels):
    # Select only from the initial posterior; do not consult held-out truth.
    candidates = np.argsort(log_weights)[-min(80, len(log_weights)):]
    residual = np.sqrt(np.mean(((prediction-observed)/NOISE)**2, axis=(1, 2)))
    best = None
    for a_index, a in enumerate(candidates):
        for b in candidates[a_index+1:]:
            disagreement = np.flatnonzero(labels[a] != labels[b])
            if not len(disagreement):
                continue
            distance = float(np.sqrt(np.mean(((prediction[a]-prediction[b])/NOISE)**2)))
            # Search for a consequential ambiguity, not merely two nearly
            # identical models lying on opposite sides of one grid boundary.
            plausible = max(residual[a], residual[b]) <= 1.3 and distance <= 0.25
            value = (-len(disagreement) + distance if plausible else
                     100 + distance + max(0, residual[a]-1.3) + max(0, residual[b]-1.3))
            if best is None or value < best[0]:
                best = (value, a, b, distance, residual[a], residual[b], disagreement)
    if best is None:
        return None
    _, a, b, distance, ra, rb, disagreement = best
    return {"indices": [int(a), int(b)], "calibration_whitened_rmse": [float(ra), float(rb)],
            "pair_whitened_rms_distance": distance,
            "both_fit_within_1_3_sigma_rms": bool(max(ra, rb) <= 1.3),
            "task_disagreement_fraction": float(len(disagreement)/labels.shape[1]),
            "selection": "largest task disagreement among plausible pairs within 0.25 noise-scaled RMS; closest-pair fallback",
            "acceleration_m_s2": float(ACCELERATIONS[disagreement[len(disagreement)//2]])}


def _write_trace(path, trajectory):
    np.savez_compressed(path, **trajectory)


def run_active_identification(output, *, hypotheses=512, trials=6, rounds=2,
                              seed=7, dt=0.0001, render=True):
    if type(trials) is not int or trials < 1 or type(rounds) is not int or not 1 <= rounds <= 4:
        raise ValueError("trials >= 1 and rounds between 1 and 4 are required")
    bank = make_bank(hypotheses, seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Active contact: {hypotheses} hypotheses, {trials} trials, {rounds} probes", flush=True)
    initial_prediction = observations(simulate(bank, CALIBRATION, dt=dt, samples=13))
    responses = []
    for probe in PROBES:
        responses.append(observations(simulate(bank, probe, dt=dt)))
        print(f"  library: {probe.name}", flush=True)
    responses = np.asarray(responses)
    labels = np.column_stack([simulate(bank, Release(float(a)), dt=dt)["detached"]
                              for a in ACCELERATIONS])
    rng = np.random.default_rng(seed+10000)
    # Truth is continuous and off-grid. It is never inserted into the hypothesis bank.
    truth_parameters = np.exp(rng.uniform(np.log(BOUNDS[:, 0]), np.log(BOUNDS[:, 1]), (trials, 4)))
    initial_truth = simulate(truth_parameters, CALIBRATION, dt=dt/2, samples=13)
    truth_responses = np.asarray([observations(simulate(truth_parameters, probe, dt=dt/2))
                                 for probe in PROBES])
    truth_labels = np.column_stack([simulate(truth_parameters, Release(float(a)), dt=dt/2)["detached"]
                                    for a in ACCELERATIONS])
    # Pre-generate observations indexed by trial/action/repetition. Selection cannot
    # access an observation before choosing its action; policies share its noise.
    initial_observed = observations(initial_truth) + rng.normal(size=observations(initial_truth).shape)*NOISE
    observed = np.transpose(truth_responses, (1, 0, 2, 3))[:, None] + rng.normal(
        size=(trials, rounds, len(PROBES), responses.shape[2], 2))*NOISE
    metrics, weights_archive, selected_archive, probability_archive = [], [], [], []
    first_trace = None
    prior = np.full(hypotheses, -np.log(hypotheses))
    for trial in range(trials):
        initial_log = log_posterior(prior, initial_prediction, initial_observed[trial])
        trial_weights, trial_actions, trial_probability = [], [], []
        for policy in POLICIES:
            log_weights = initial_log.copy()
            history, actions, probabilities = [], [], []
            for step in range(rounds+1):
                metrics.append({"trial": trial, "policy": policy, "step": step,
                                **_metrics(log_weights, labels, truth_labels[trial])})
                history.append(np.exp(log_weights))
                probabilities.append(np.exp(log_weights)@labels)
                if step == rounds:
                    break
                if policy in ("task", "information"):
                    score = acquisition_scores(log_weights, responses, labels, objective=policy,
                                               seed=seed+1000*trial+step)
                    action = int(np.argmax(score))
                elif policy == "fixed":
                    # Predeclared multirate schedule, independent of results.
                    action = (5, 4, 7, 6)[step]
                else:
                    action = int(np.random.default_rng(seed+3000*trial+step).integers(len(PROBES)))
                actions.append(action)
                log_weights = log_posterior(log_weights, responses[action], observed[trial, step, action])
            trial_weights.append(history)
            trial_actions.append(actions)
            trial_probability.append(probabilities)
        weights_archive.append(trial_weights)
        selected_archive.append(trial_actions)
        probability_archive.append(trial_probability)
        if trial == 0:
            pair = _ambiguous_pair(initial_log, initial_prediction, initial_observed[0], labels)
            first_trace = (pair, trial_actions[0])
        print(f"  evaluated trial {trial+1}/{trials}", flush=True)

    summary = {}
    for policy in POLICIES:
        before = [m for m in metrics if m["policy"] == policy and m["step"] == 0]
        after = [m for m in metrics if m["policy"] == policy and m["step"] == rounds]
        delta = np.array([a["brier"]-b["brier"] for a, b in zip(after, before)])
        summary[policy] = {
            "initial_brier_mean": float(np.mean([m["brier"] for m in before])),
            "final_brier_mean": float(np.mean([m["brier"] for m in after])),
            "paired_brier_change_mean": float(delta.mean()),
            "paired_brier_change_standard_error": (float(delta.std(ddof=1)/np.sqrt(trials)) if trials > 1 else None),
            "final_accuracy_mean": float(np.mean([m["accuracy"] for m in after])),
            "median_effective_hypotheses": float(np.median([m["effective_hypotheses"] for m in after])),
            "degenerate_trial_fraction": float(np.mean([m["finite_bank_degenerate"] for m in after]))}

    # Independent integration checks at a weak probe, a nonlinear probe and release.
    checks = []
    for protocol in (CALIBRATION, PROBES[-1], Release(52.0)):
        coarse = simulate(truth_parameters[0], protocol, dt=dt)
        fine = simulate(truth_parameters[0], protocol, dt=dt/2)
        radau = simulate(truth_parameters[0], protocol, method="radau")
        checks.append({"protocol": asdict(protocol),
                       "rk4_half_dt_force_max_error_n": float(np.max(np.abs(coarse["force_n"]-fine["force_n"]))),
                       "rk4_radau_force_max_error_n": float(np.max(np.abs(coarse["force_n"]-radau["force_n"]))),
                       "rk4_radau_position_max_error_m": float(np.max(np.abs(coarse["state"][:, :, [0, 2]]-radau["state"][:, :, [0, 2]]))),
                       "detachment_agrees": bool(np.array_equal(coarse["detached"], radau["detached"]))})
    pair, chosen = first_trace
    if pair is not None:
        pair["parameters"] = bank[pair["indices"]].tolist()
        pair_trajectory = simulate(bank[pair["indices"]], Release(pair["acceleration_m_s2"]), dt=dt/2, samples=91)
        pair["release_detached"] = pair_trajectory["detached"].tolist()
        pair["prediction_disagreement_verified_at_half_dt"] = bool(pair_trajectory["detached"][0] != pair_trajectory["detached"][1])
        _write_trace(output/"ambiguity_release.npz", pair_trajectory)
        _write_trace(output/"ambiguity_probe.npz", simulate(bank[pair["indices"]], CALIBRATION, dt=dt/2, samples=61))
    for i, action in enumerate(chosen):
        _write_trace(output/f"probe_{i+1}.npz", simulate(truth_parameters[0], PROBES[action], dt=dt/2, samples=91))
    _write_trace(output/"initial_probe.npz", simulate(truth_parameters[0], CALIBRATION, dt=dt/2, samples=61))

    task_brier = np.array([m["brier"] for m in metrics if m["policy"] == "task" and m["step"] == rounds])
    fixed_brier = np.array([m["brier"] for m in metrics if m["policy"] == "fixed" and m["step"] == rounds])
    report = {
        "model": "Axial lumped-mass SLS + reversible triangular adhesive potential",
        "seed": seed, "hypotheses": hypotheses, "trials": trials, "rounds": rounds,
        "spec": asdict(ContactSpec()), "dt_s": dt,
        "parameter_columns": ["equilibrium_pa", "branch_pa", "relaxation_s", "work_j_m2"],
        "log_uniform_parameter_bounds": BOUNDS.tolist(),
        "observation_noise_std": {"force_n": float(NOISE[0]), "marker_m": float(NOISE[1])},
        "calibration": asdict(CALIBRATION), "probes": [asdict(p) for p in PROBES],
        "release_acceleration_m_s2": ACCELERATIONS.tolist(),
        "release_definition": "gap >= cutoff at every observation in the final 20 ms",
        "policies": summary, "ambiguity_pair": pair, "integration_checks": checks,
        "task_minus_fixed_brier": {
            "paired_mean": float(np.mean(task_brier-fixed_brier)),
            "paired_standard_error": (float(np.std(task_brier-fixed_brier, ddof=1)/np.sqrt(trials)) if trials > 1 else None),
            "interpretation": "negative favors task; a pilot, not a significance claim"},
        "first_trial_task_probes": [PROBES[i].name for i in chosen],
        "code_sha256": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                        for name in ("active_contact.py", "active_identification.py")},
        "limitations": [
            "Numerical proof of concept; no experimental or 3D continuum validation.",
            "Truth is off-grid and uses half the RK4 time step, but shares the inference constitutive law.",
            "Radau verifies integration only; no independent physics model or model-mismatch validation yet.",
            "Finite hypothesis posterior can collapse or exclude truth; ESS is reported, intervals are not calibrated.",
            "Every probe starts on an intact, fully rested specimen; irreversible damage/history between probes is absent.",
            "Only normal axial motion, known geometry/masses, fixed interface range and force/one marker sensing.",
            "Equal duration/samples per additional probe; force exposure and damage costs are not constrained.",
            "Monte Carlo acquisition and a small number of trials do not establish a universal policy advantage."]}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    import csv
    with (output/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    np.savez_compressed(output/"benchmark.npz", parameters=bank, truth_parameters=truth_parameters,
                        initial_prediction=initial_prediction, initial_observed=initial_observed,
                        probe_predictions=responses, observed=observed, task_labels=labels,
                        truth_labels=truth_labels, posterior_weights=weights_archive,
                        chosen_actions=selected_archive, release_probability=probability_archive,
                        accelerations_m_s2=ACCELERATIONS, policies=POLICIES)
    if render:
        from .active_render import render_active_identification
        render_active_identification(output)
    return report
