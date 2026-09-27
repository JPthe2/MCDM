import numpy as np
import pandas as pd
import pytest

from core.models import MCDMError
from methods.entropy import entropy_weights


def manual_entropy(matrix: np.ndarray, benefit: list[bool]) -> np.ndarray:
    lo, hi = matrix.min(axis=0), matrix.max(axis=0)
    r = np.where(benefit, (matrix - lo) / (hi - lo), (hi - matrix) / (hi - lo))
    p = r / r.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = -np.nansum(np.where(p > 0, p * np.log(p), 0), axis=0) / np.log(len(matrix))
    d = 1 - e
    return d / d.sum()


def test_matches_manual_calculation(small_matrix, small_types):
    result = entropy_weights(small_matrix, small_types)
    expected = manual_entropy(small_matrix.to_numpy(), [t == "benefit" for t in small_types])
    np.testing.assert_allclose(result.weights.values, expected, atol=1e-12)
    assert result.weights.sum() == pytest.approx(1.0)
    assert (result.weights >= 0).all()
    summary = result.details["summary"]
    assert ((summary["Entropy e_j"] >= 0) & (summary["Entropy e_j"] <= 1)).all()


def test_cost_criterion_is_mirror_of_benefit():
    benefit = pd.DataFrame({"x": [1.0, 4.0, 2.0, 8.0], "y": [3.0, 1.0, 2.0, 5.0]})
    mirrored = pd.DataFrame({"x": [-1.0, -4.0, -2.0, -8.0], "y": [3.0, 1.0, 2.0, 5.0]})
    a = entropy_weights(benefit, ["benefit", "benefit"])
    b = entropy_weights(mirrored, ["cost", "benefit"])
    np.testing.assert_allclose(a.weights.values, b.weights.values, atol=1e-12)


def test_constant_column_gets_zero_weight(small_matrix, small_types):
    matrix = small_matrix.copy()
    matrix["Quality"] = 7.0
    result = entropy_weights(matrix, small_types)
    assert result.weights["Quality"] == pytest.approx(0.0, abs=1e-12)
    assert result.weights.sum() == pytest.approx(1.0)
    assert any("Constant" in w for w in result.warnings)


def test_zero_values_are_handled(small_types):
    matrix = pd.DataFrame([[0, 0, 5, 0], [1, 2, 0, 3], [2, 0, 1, 1], [0, 1, 2, 0]], dtype=float)
    result = entropy_weights(matrix, small_types)
    assert np.all(np.isfinite(result.weights.values))
    assert result.weights.sum() == pytest.approx(1.0)


def test_all_constant_criteria_raise():
    matrix = pd.DataFrame([[1, 5], [1, 5], [1, 5]], dtype=float)
    with pytest.raises(MCDMError, match="zero divergence"):
        entropy_weights(matrix, ["benefit", "cost"])


def test_single_alternative_raises():
    with pytest.raises(MCDMError, match="two alternatives"):
        entropy_weights(pd.DataFrame([[1.0, 2.0]]), ["benefit", "benefit"])


def test_missing_value_raises(small_matrix, small_types):
    matrix = small_matrix.copy()
    matrix.iloc[1, 2] = np.nan
    with pytest.raises(MCDMError, match="missing"):
        entropy_weights(matrix, small_types)


def test_reproducible(small_matrix, small_types):
    a = entropy_weights(small_matrix, small_types).weights
    b = entropy_weights(small_matrix, small_types).weights
    pd.testing.assert_series_equal(a, b)
