import numpy as np
import pytest

from core.models import MCDMError
from methods.ahp import (RANDOM_INDEX, ahp_weights, build_pairwise_matrix, random_index,
                         validate_pairwise_matrix)


def consistent_matrix(weights: np.ndarray) -> np.ndarray:
    return weights[:, None] / weights[None, :]


def test_consistent_matrix_recovers_weights_and_zero_cr():
    w = np.array([0.4, 0.3, 0.2, 0.1])
    result = ahp_weights(consistent_matrix(w), ["a", "b", "c", "d"])
    np.testing.assert_allclose(result.weights.values, w, atol=1e-10)
    assert result.details["lambda_max"] == pytest.approx(4.0, abs=1e-9)
    assert result.details["CI"] == pytest.approx(0.0, abs=1e-9)
    assert result.details["CR"] == pytest.approx(0.0, abs=1e-9)
    assert result.details["consistent"]
    assert result.warnings == []


def test_known_saaty_example():
    # Widely used 3x3 example: weights ≈ (0.637, 0.258, 0.105), λmax ≈ 3.0385.
    a = np.array([[1, 3, 5], [1 / 3, 1, 3], [1 / 5, 1 / 3, 1]])
    result = ahp_weights(a)
    np.testing.assert_allclose(result.weights.values, [0.6370, 0.2583, 0.1047], atol=1e-4)
    assert result.details["lambda_max"] == pytest.approx(3.0385, abs=1e-4)
    assert result.details["CI"] == pytest.approx((3.0385 - 3) / 2, abs=1e-4)
    assert result.details["CR"] == pytest.approx(result.details["CI"] / 0.58, rel=1e-9)
    assert result.details["CR"] < 0.10


def test_weights_normalised_and_nonnegative():
    a = build_pairwise_matrix(5, {(0, 1): 3, (0, 2): 7, (1, 3): 1 / 5, (2, 4): 9, (3, 4): 2})
    result = ahp_weights(a)
    assert result.weights.sum() == pytest.approx(1.0)
    assert (result.weights >= 0).all()


def test_inconsistent_matrix_warns():
    # a > b (9), b > c (9) but c > a (9): strongly intransitive.
    a = build_pairwise_matrix(3, {(0, 1): 9, (1, 2): 9, (0, 2): 1 / 9})
    result = ahp_weights(a)
    assert result.details["CR"] > 0.10
    assert not result.details["consistent"]
    assert any("exceeds" in w for w in result.warnings)


def test_build_pairwise_matrix_is_reciprocal_with_unit_diagonal():
    a = build_pairwise_matrix(4, {(0, 1): 3, (2, 1): 5, (0, 3): 1 / 7})
    np.testing.assert_allclose(np.diag(a), 1.0)
    np.testing.assert_allclose(a * a.T, 1.0)
    assert a[1, 2] == pytest.approx(1 / 5)
    assert a[3, 0] == pytest.approx(7)


def test_single_criterion():
    result = ahp_weights(np.array([[1.0]]), ["only"])
    assert result.weights.tolist() == [1.0]
    assert result.details["CR"] == 0.0


def test_two_criteria_always_consistent():
    result = ahp_weights(build_pairwise_matrix(2, {(0, 1): 4}))
    np.testing.assert_allclose(result.weights.values, [0.8, 0.2])
    assert result.details["CR"] == 0.0


@pytest.mark.parametrize("matrix, message", [
    (np.array([[1, 2], [0.4, 1]]), "reciprocal"),
    (np.array([[2, 2], [0.5, 1]]), "diagonal"),
    (np.array([[1, -2], [-0.5, 1]]), "positive"),
    (np.array([[1, 12], [1 / 12, 1]]), "Saaty"),
    (np.ones((2, 3)), "square"),
])
def test_invalid_matrices_rejected(matrix, message):
    with pytest.raises(MCDMError, match=message):
        validate_pairwise_matrix(matrix)


def test_invalid_judgment_value_rejected():
    with pytest.raises(MCDMError):
        build_pairwise_matrix(3, {(0, 1): 0})
    with pytest.raises(MCDMError):
        build_pairwise_matrix(3, {(0, 0): 2})


def test_random_index_table_and_limit():
    assert random_index(3) == RANDOM_INDEX[3] == 0.58
    assert random_index(10) == 1.49
    with pytest.raises(MCDMError):
        random_index(16)


def test_reproducible():
    a = build_pairwise_matrix(4, {(0, 1): 3, (0, 2): 5, (1, 2): 2, (2, 3): 3})
    first, second = ahp_weights(a), ahp_weights(a)
    np.testing.assert_array_equal(first.weights.values, second.weights.values)
