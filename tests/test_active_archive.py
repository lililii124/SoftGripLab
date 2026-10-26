"""Audit the saved benchmark for fair observations and reproducible metrics."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp


ARCHIVE = Path(__file__).resolve().parents[1]/"docs/results/active_identification"


@pytest.fixture(scope="module")
def archive():
    if not (ARCHIVE/"benchmark.npz").exists():
        pytest.skip("archived active-identification benchmark unavailable")
    with np.load(ARCHIVE/"benchmark.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return arrays, json.loads((ARCHIVE/"report.json").read_text(encoding="utf-8"))


def test_archive_policies_only_condition_on_selected_observations(archive):
    data, report = archive
    noise = np.array([report["observation_noise_std"]["force_n"],
                      report["observation_noise_std"]["marker_m"]])
    for trial in range(report["trials"]):
        assert np.all(np.any(data["parameters"] != data["truth_parameters"][trial], axis=1))
        initial_residual = (data["initial_prediction"]-data["initial_observed"][trial])/noise
        initial = -0.5*np.sum(initial_residual**2, axis=(1, 2))
        for policy in range(len(data["policies"])):
            log_weight = initial-logsumexp(initial)
            np.testing.assert_allclose(data["posterior_weights"][trial, policy, 0], np.exp(log_weight))
            for step, action in enumerate(data["chosen_actions"][trial, policy]):
                residual = (data["probe_predictions"][action]-data["observed"][trial, step, action])/noise
                log_weight -= 0.5*np.sum(residual**2, axis=(1, 2))
                log_weight -= logsumexp(log_weight)
                np.testing.assert_allclose(data["posterior_weights"][trial, policy, step+1], np.exp(log_weight), atol=1e-12)


def test_archive_probability_and_scores_match_saved_truth(archive):
    data, report = archive
    expected = data["posterior_weights"]@data["task_labels"].astype(float)
    np.testing.assert_allclose(data["release_probability"], expected, atol=1e-12)
    with (ARCHIVE/"metrics.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            trial, step = int(row["trial"]), int(row["step"])
            policy = list(data["policies"]).index(row["policy"])
            probability = expected[trial, policy, step]
            score = np.mean((probability-data["truth_labels"][trial])**2)
            assert float(row["brier"]) == pytest.approx(score, abs=1e-14)
    pair = report["ambiguity_pair"]
    assert pair is not None and pair["prediction_disagreement_verified_at_half_dt"]
    assert pair["both_fit_within_1_3_sigma_rms"]
    assert len(set(pair["release_detached"])) == 2
    for check in report["integration_checks"]:
        assert check["rk4_radau_force_max_error_n"] < report["observation_noise_std"]["force_n"]/10
        assert check["rk4_radau_position_max_error_m"] < report["observation_noise_std"]["marker_m"]/10
        assert check["detachment_agrees"]
