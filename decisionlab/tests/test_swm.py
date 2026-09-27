import numpy as np
import pandas as pd
import pytest

from core.models import MCDMError
from methods.swm import swm


def test_hand_computed_scores(small_matrix, small_types):
    # Cost: 90..120, Quality: 6..9, Time: 3..7, Reliability: 0.80..0.95
    w = [0.4, 0.3, 0.2, 0.1]
    result = swm(small_matrix, small_types, w)
    expected = {
        "S1": 0.4 * (120 - 100) / 30 + 0.3 * (8 - 6) / 3 + 0.2 * (7 - 5) / 4 + 0.1 * (0.90 - 0.80) / 0.15,
        "S2": 0.4 * 0 + 0.3 * 1 + 0.2 * 1 + 0.1 * 1,
        "S3": 0.4 * 1 + 0.3 * 0 + 0.2 * 0 + 0.1 * 0,
        "S4": 0.4 * (10 / 30) + 0.3 * (1 / 3) + 0.2 * (3 / 4) + 0.1 * (0.05 / 0.15),
    }
    for alt, score in expected.items():
        assert result.scores[alt] == pytest.approx(score, abs=1e-12)
    assert result.ranking["Alternative"].tolist() == ["S1", "S2", "S4", "S3"]


def test_contributions_sum_to_score(small_matrix, small_types):
    result = swm(small_matrix, small_types, [0.25] * 4)
    np.testing.assert_allclose(result.details["weighted_matrix"].sum(axis=1), result.scores)
    norm = result.details["normalised_matrix"]
    assert norm.min().min() >= 0 and norm.max().max() <= 1


def test_cost_criterion_lower_is_better():
    matrix = pd.DataFrame({"price": [10.0, 20.0, 30.0]}, index=["cheap", "mid", "pricey"])
    result = swm(matrix, ["cost"], [1.0])
    assert result.scores.tolist() == [1.0, 0.5, 0.0]
    assert result.ranking.iloc[0]["Alternative"] == "cheap"


def test_negative_values():
    matrix = pd.DataFrame({"profit": [-5.0, 0.0, 5.0], "risk": [-1.0, -3.0, -2.0]},
                          index=["a", "b", "c"])
    result = swm(matrix, ["benefit", "cost"], [0.5, 0.5])
    assert result.scores["a"] == pytest.approx(0.5 * 0 + 0.5 * 0)
    assert result.scores["b"] == pytest.approx(0.5 * 0.5 + 0.5 * 1)
    assert result.scores["c"] == pytest.approx(0.5 * 1 + 0.5 * 0.5)


def test_constant_column_does_not_change_order(small_matrix, small_types):
    w = [0.4, 0.3, 0.2, 0.1]
    base = swm(small_matrix, small_types, w)
    matrix = small_matrix.assign(Extra=3.0)
    extended = swm(matrix, small_types + ["benefit"], [0.3, 0.3, 0.2, 0.1, 0.1])
    assert np.isfinite(extended.scores).all()
    assert any("Constant" in msg for msg in extended.warnings)
    no_constant = swm(small_matrix, small_types, np.array([0.3, 0.3, 0.2, 0.1]) / 0.9)
    assert extended.ranking["Alternative"].tolist() == no_constant.ranking["Alternative"].tolist()
    assert len(base.ranking) == 4


def test_ties_share_rank():
    matrix = pd.DataFrame({"x": [1.0, 2.0, 2.0, 0.0]}, index=["a", "b", "c", "d"])
    result = swm(matrix, ["benefit"], [1.0])
    assert result.ranking.set_index("Alternative")["Rank"].to_dict() == \
        {"b": 1, "c": 1, "a": 3, "d": 4}


def test_invalid_inputs(small_matrix, small_types):
    with pytest.raises(MCDMError, match="sum to 1"):
        swm(small_matrix, small_types, [0.3, 0.3, 0.3, 0.3])
    with pytest.raises(MCDMError, match="Expected 4 criterion types"):
        swm(small_matrix, small_types[:3], [0.25] * 4)
    bad = small_matrix.copy()
    bad.iloc[2, 1] = np.nan
    with pytest.raises(MCDMError, match="missing"):
        swm(bad, small_types, [0.25] * 4)


def test_reproducible(small_matrix, small_types):
    a = swm(small_matrix, small_types, [0.4, 0.3, 0.2, 0.1])
    b = swm(small_matrix, small_types, [0.4, 0.3, 0.2, 0.1])
    pd.testing.assert_frame_equal(a.ranking, b.ranking)
