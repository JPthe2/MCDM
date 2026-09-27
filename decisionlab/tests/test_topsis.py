import numpy as np
import pandas as pd
import pytest

from core.models import MCDMError
from methods.topsis import topsis


def test_known_example(phone_matrix, phone_types):
    # Classic textbook example with equal weights.
    result = topsis(phone_matrix, phone_types, [0.25] * 4)
    expected = {"A": 0.5343, "B": 0.3084, "C": 0.6916, "D": 0.5347, "E": 0.4010}
    for alt, cc in expected.items():
        assert result.scores[alt] == pytest.approx(cc, abs=1e-3)
    assert result.ranking["Alternative"].tolist() == ["C", "D", "A", "E", "B"]
    assert result.ranking["Rank"].tolist() == [1, 2, 3, 4, 5]


def test_intermediate_matrices(phone_matrix, phone_types):
    w = np.array([0.4, 0.2, 0.3, 0.1])
    result = topsis(phone_matrix, phone_types, w)
    norm = result.details["normalised_matrix"]
    np.testing.assert_allclose((norm ** 2).sum(axis=0).values, 1.0)
    np.testing.assert_allclose(result.details["weighted_matrix"].values, norm.values * w)
    ideal = result.details["ideal_solutions"]
    weighted = result.details["weighted_matrix"]
    # Price is a cost criterion: its ideal best is the minimum.
    assert ideal.loc["Ideal best (A+)", "Price"] == pytest.approx(weighted["Price"].min())
    assert ideal.loc["Ideal worst (A−)", "Price"] == pytest.approx(weighted["Price"].max())
    assert ideal.loc["Ideal best (A+)", "Camera"] == pytest.approx(weighted["Camera"].max())
    d = result.details["distances"]
    np.testing.assert_allclose(d["Closeness CC"], d.iloc[:, 1] / (d.iloc[:, 0] + d.iloc[:, 1]))


def test_dominant_alternative_ranks_first(small_matrix, small_types):
    matrix = small_matrix.copy()
    matrix.loc["Best"] = [80, 10, 2, 0.99]  # cheapest, best quality, fastest, most reliable
    result = topsis(matrix, small_types, [0.25] * 4)
    assert result.ranking.iloc[0]["Alternative"] == "Best"
    assert result.scores["Best"] == pytest.approx(1.0)


def test_benefit_and_cost_reverse_order():
    matrix = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=["low", "mid", "high"])
    as_benefit = topsis(matrix, ["benefit"], [1.0]).ranking["Alternative"].tolist()
    as_cost = topsis(matrix, ["cost"], [1.0]).ranking["Alternative"].tolist()
    assert as_benefit == ["high", "mid", "low"]
    assert as_cost == ["low", "mid", "high"]


def test_scores_in_unit_interval(small_matrix, small_types):
    result = topsis(small_matrix, small_types, [0.1, 0.4, 0.3, 0.2])
    assert ((result.scores >= 0) & (result.scores <= 1)).all()


def test_zero_and_constant_columns(small_matrix, small_types):
    matrix = small_matrix.copy()
    matrix["Quality"] = 0.0
    matrix["Time"] = 5.0
    result = topsis(matrix, small_types, [0.25] * 4)
    assert np.all(np.isfinite(result.scores))
    assert any("All-zero" in w for w in result.warnings)


def test_identical_alternatives_tie():
    matrix = pd.DataFrame([[1.0, 2.0], [1.0, 2.0]], index=["a", "b"], columns=["x", "y"])
    result = topsis(matrix, ["benefit", "cost"], [0.5, 0.5])
    assert result.scores.tolist() == [0.5, 0.5]
    assert result.ranking["Rank"].tolist() == [1, 1]
    assert result.warnings


def test_single_alternative():
    result = topsis(pd.DataFrame({"x": [3.0], "y": [1.0]}), ["benefit", "cost"], [0.5, 0.5])
    assert result.ranking["Rank"].tolist() == [1]


@pytest.mark.parametrize("weights, message", [
    ([0.5, 0.5, 0.5, 0.5], "sum to 1"),
    ([1.2, -0.2, 0.0, 0.0], "non-negative"),
    ([0.5, 0.5], "Expected 4 weights"),
    ([np.nan, 0.5, 0.25, 0.25], "missing"),
])
def test_invalid_weights_rejected(small_matrix, small_types, weights, message):
    with pytest.raises(MCDMError, match=message):
        topsis(small_matrix, small_types, weights)


def test_invalid_type_rejected(small_matrix):
    with pytest.raises(MCDMError, match="Benefit' or 'Cost"):
        topsis(small_matrix, ["cost", "good", "cost", "benefit"], [0.25] * 4)


def test_weights_series_aligned_by_name(phone_matrix, phone_types):
    w = pd.Series({"Looks": 0.1, "Camera": 0.3, "Storage": 0.2, "Price": 0.4})
    by_name = topsis(phone_matrix, phone_types, w)
    by_position = topsis(phone_matrix, phone_types, [0.4, 0.2, 0.3, 0.1])
    pd.testing.assert_series_equal(by_name.scores, by_position.scores)


def test_reproducible(phone_matrix, phone_types):
    a = topsis(phone_matrix, phone_types, [0.4, 0.2, 0.3, 0.1])
    b = topsis(phone_matrix, phone_types, [0.4, 0.2, 0.3, 0.1])
    pd.testing.assert_frame_equal(a.ranking, b.ranking)
