import numpy as np
import pytest

from core.models import MCDMError
from methods.bwm import bwm_weights


def constraint_violation(result, best, worst, a_b, a_w):
    w = result.weights.to_numpy()
    xi = result.details["xi"]
    dev_b = np.abs(w[best] / w - np.asarray(a_b))
    dev_w = np.abs(w / w[worst] - np.asarray(a_w))
    return max(dev_b.max(), dev_w.max()) - xi


def test_fully_consistent_comparisons_give_exact_weights():
    # Weights proportional to 8:4:2:1 make every comparison exact.
    result = bwm_weights(0, 3, [1, 2, 4, 8], [8, 4, 2, 1])
    np.testing.assert_allclose(result.weights.values, np.array([8, 4, 2, 1]) / 15, atol=1e-6)
    assert result.details["xi"] == pytest.approx(0.0, abs=1e-6)
    assert result.details["solver_success"]


def test_inconsistent_input_satisfies_model_constraints():
    best, worst = 0, 4
    a_b = [1, 2, 4, 3, 8]
    a_w = [8, 5, 2, 3, 1]
    result = bwm_weights(best, worst, a_b, a_w)
    w = result.weights
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert (w >= 0).all()
    assert result.details["xi"] > 0
    assert constraint_violation(result, best, worst, a_b, a_w) <= 1e-6
    assert w.idxmax() == w.index[best]
    assert w.idxmin() == w.index[worst]
    assert 0 <= result.details["consistency_ratio"] < 1


def test_two_criteria_ratio_matches_judgment():
    result = bwm_weights(1, 0, [3, 1], [1, 3], ["low", "high"])
    assert result.weights["high"] / result.weights["low"] == pytest.approx(3.0, rel=1e-5)


def test_single_criterion():
    result = bwm_weights(0, 0, [1], [1], ["only"])
    assert result.weights.tolist() == [1.0]


@pytest.mark.parametrize("kwargs, message", [
    (dict(best=0, worst=0, best_to_others=[1, 2, 3], others_to_worst=[3, 2, 1]), "different"),
    (dict(best=0, worst=2, best_to_others=[2, 2, 3], others_to_worst=[3, 2, 1]), "itself"),
    (dict(best=0, worst=2, best_to_others=[1, 2, 3], others_to_worst=[3, 2, 2]), "itself"),
    (dict(best=0, worst=2, best_to_others=[1, 2, 12], others_to_worst=[12, 2, 1]), "1–9"),
    (dict(best=0, worst=2, best_to_others=[1, 2, 5], others_to_worst=[4, 2, 1]), "same"),
    (dict(best=0, worst=5, best_to_others=[1, 2, 5], others_to_worst=[5, 2, 1]), "valid"),
    (dict(best=0, worst=2, best_to_others=[1, 2], others_to_worst=[5, 2, 1]), "one value"),
])
def test_invalid_inputs_rejected(kwargs, message):
    with pytest.raises(MCDMError, match=message):
        bwm_weights(**kwargs)


def test_ordinal_inconsistency_is_flagged():
    # Best is preferred over criterion 1 (9) more than over the worst (3): contradictory.
    result = bwm_weights(0, 2, [1, 9, 3], [3, 2, 1])
    assert any("inconsistent" in w for w in result.warnings)


def test_reproducible():
    args = (1, 3, [3, 1, 2, 6], [2, 6, 3, 1])
    first, second = bwm_weights(*args), bwm_weights(*args)
    np.testing.assert_allclose(first.weights.values, second.weights.values, atol=1e-12)
    assert first.details["xi"] == second.details["xi"]
