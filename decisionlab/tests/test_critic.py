import numpy as np
import pandas as pd
import pytest

from core.models import MCDMError
from methods.critic import critic_weights


def manual_critic(matrix: np.ndarray, benefit: list[bool]) -> np.ndarray:
    lo, hi = matrix.min(axis=0), matrix.max(axis=0)
    x = np.where(benefit, (matrix - lo) / (hi - lo), (hi - matrix) / (hi - lo))
    sigma = x.std(axis=0, ddof=1)
    corr = np.corrcoef(x, rowvar=False)
    c = sigma * (1 - corr).sum(axis=1)
    return c / c.sum()


def test_matches_manual_calculation(small_matrix, small_types):
    result = critic_weights(small_matrix, small_types)
    expected = manual_critic(small_matrix.to_numpy(), [t == "benefit" for t in small_types])
    np.testing.assert_allclose(result.weights.values, expected, atol=1e-12)
    assert result.weights.sum() == pytest.approx(1.0)
    assert (result.weights >= 0).all()


def test_ddof_choice_does_not_change_weights(small_matrix, small_types):
    # σ with ddof=0 or ddof=1 differs by a constant factor that cancels out.
    x = small_matrix.to_numpy()
    benefit = [t == "benefit" for t in small_types]
    lo, hi = x.min(axis=0), x.max(axis=0)
    r = np.where(benefit, (x - lo) / (hi - lo), (hi - x) / (hi - lo))
    c = r.std(axis=0, ddof=0) * (1 - np.corrcoef(r, rowvar=False)).sum(axis=1)
    np.testing.assert_allclose(critic_weights(small_matrix, small_types).weights.values,
                               c / c.sum(), atol=1e-12)


def test_constant_column_gets_zero_weight(small_matrix, small_types):
    matrix = small_matrix.copy()
    matrix["Time"] = 4.0
    result = critic_weights(matrix, small_types)
    assert result.weights["Time"] == 0.0
    assert result.weights.sum() == pytest.approx(1.0)
    corr = result.details["correlation_matrix"]
    assert corr.loc["Time", "Time"] == 1.0
    assert (corr.loc["Time"].drop("Time") == 0.0).all()


def test_duplicate_criteria_share_information():
    base = pd.DataFrame({"a": [1.0, 5.0, 3.0, 2.0], "b": [4.0, 1.0, 2.0, 5.0]})
    duplicated = base.assign(a_copy=base["a"])
    w1 = critic_weights(base, ["benefit"] * 2).weights
    w2 = critic_weights(duplicated, ["benefit"] * 3).weights
    assert w2["a"] == pytest.approx(w2["a_copy"])
    # Duplicating 'a' must not simply double its influence relative to 'b'.
    assert (w2["a"] + w2["a_copy"]) / w2["b"] < 2 * w1["a"] / w1["b"]


def test_cost_type_changes_direction_not_information():
    matrix = pd.DataFrame({"a": [1.0, 5.0, 3.0, 2.0], "b": [4.0, 1.0, 2.0, 5.0],
                           "c": [2.0, 2.5, 9.0, 1.0]})
    w_benefit = critic_weights(matrix, ["benefit", "benefit", "benefit"]).weights
    w_cost = critic_weights(matrix.assign(a=-matrix["a"]), ["cost", "benefit", "benefit"]).weights
    np.testing.assert_allclose(w_benefit.values, w_cost.values, atol=1e-12)


def test_all_constant_raises():
    with pytest.raises(MCDMError, match="zero information"):
        critic_weights(pd.DataFrame([[1, 2], [1, 2], [1, 2]], dtype=float), ["benefit", "cost"])


def test_single_criterion_and_single_alternative():
    single = critic_weights(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), ["benefit"])
    assert single.weights.tolist() == [1.0]
    with pytest.raises(MCDMError, match="two alternatives"):
        critic_weights(pd.DataFrame([[1.0, 2.0]]), ["benefit", "benefit"])


def test_non_numeric_raises(small_matrix, small_types):
    matrix = small_matrix.astype(object)
    matrix.iloc[0, 0] = "abc"
    with pytest.raises(MCDMError, match="non-numeric"):
        critic_weights(matrix, small_types)
