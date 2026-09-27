import numpy as np
import pandas as pd
import pytest

from core.explain import explain_ranking
from core.models import MCDMError
from core.normalization import min_max_normalize, vector_normalize
from core.validation import (normalise_weight_vector, validate_criteria_types,
                             validate_decision_matrix, validate_names, validate_weights)
from methods import ahp_weights, build_pairwise_matrix, swm, topsis
from utils.csv_io import matrix_to_csv, parse_matrix_csv, template_csv
from utils.export import ReportData, excel_report, pdf_available, pdf_report


# ---------------------------------------------------------------- decision matrix

def test_valid_matrix_is_converted_to_float(small_matrix):
    frame = validate_decision_matrix(small_matrix.astype(object))
    assert frame.dtypes.eq(float).all()


def test_array_input_gets_default_labels():
    frame = validate_decision_matrix([[1, 2], [3, 4]])
    assert list(frame.index) == ["A1", "A2"] and list(frame.columns) == ["C1", "C2"]


def test_missing_and_non_numeric_values_are_reported(small_matrix):
    matrix = small_matrix.astype(object)
    matrix.iloc[0, 1] = None
    matrix.iloc[2, 3] = "high"
    with pytest.raises(MCDMError) as info:
        validate_decision_matrix(matrix)
    message = str(info.value)
    assert "2 missing or non-numeric" in message
    assert "S1 / Quality" in message and "S3 / Reliability" in message


def test_infinite_values_rejected():
    with pytest.raises(MCDMError):
        validate_decision_matrix(pd.DataFrame([[1.0, np.inf]]))


def test_empty_matrix_rejected():
    with pytest.raises(MCDMError, match="empty"):
        validate_decision_matrix(pd.DataFrame())


def test_duplicate_and_blank_names_rejected():
    with pytest.raises(MCDMError, match="unique"):
        validate_names(["a", "b", "a"], "alternative")
    with pytest.raises(MCDMError, match="needs a name"):
        validate_names(["a", " ", None], "criterion")
    with pytest.raises(MCDMError, match="unique"):
        validate_decision_matrix(pd.DataFrame([[1, 2]], columns=["x", "x"]))


def test_criteria_types():
    assert validate_criteria_types(["Benefit", " COST "], 2) == ["benefit", "cost"]
    with pytest.raises(MCDMError):
        validate_criteria_types(["benefit", "maybe"], 2)
    with pytest.raises(MCDMError):
        validate_criteria_types(["benefit"], 2)


# ------------------------------------------------------------------------ weights

def test_weights_validation():
    np.testing.assert_allclose(validate_weights([0.2, 0.3, 0.5], 3), [0.2, 0.3, 0.5])
    with pytest.raises(MCDMError, match="sum to 1"):
        validate_weights([0.2, 0.3, 0.6], 3)
    with pytest.raises(MCDMError, match="non-negative"):
        validate_weights([1.5, -0.5], 2)
    with pytest.raises(MCDMError, match="Expected 3"):
        validate_weights([0.5, 0.5], 3)
    # Within numerical tolerance is accepted.
    validate_weights([1 / 3, 1 / 3, 1 / 3 + 1e-9], 3)


def test_normalise_weight_vector():
    np.testing.assert_allclose(normalise_weight_vector([2, 2, 4]), [0.25, 0.25, 0.5])
    with pytest.raises(MCDMError):
        normalise_weight_vector([0, 0])


# ------------------------------------------------------------------ normalisation

def test_min_max_normalisation_directions_and_constant():
    frame = pd.DataFrame({"b": [1.0, 3.0, 2.0], "c": [1.0, 3.0, 2.0], "k": [5.0, 5.0, 5.0]})
    out = min_max_normalize(frame, ["benefit", "cost", "benefit"])
    assert out["b"].tolist() == [0.0, 1.0, 0.5]
    assert out["c"].tolist() == [1.0, 0.0, 0.5]
    assert out["k"].tolist() == [1.0, 1.0, 1.0]


def test_vector_normalisation_zero_column():
    out = vector_normalize(pd.DataFrame({"x": [3.0, 4.0], "z": [0.0, 0.0]}))
    assert out["x"].tolist() == [0.6, 0.8]
    assert out["z"].tolist() == [0.0, 0.0]


# ---------------------------------------------------------------------------- CSV

def test_csv_roundtrip_with_types(small_matrix, small_types):
    data = matrix_to_csv(small_matrix, [t.capitalize() for t in small_types])
    parsed = parse_matrix_csv(data)
    pd.testing.assert_frame_equal(parsed.matrix, small_matrix, check_names=False)
    assert parsed.types == ["Cost", "Benefit", "Cost", "Benefit"]
    assert parsed.invalid_cells == []


def test_csv_semicolon_and_invalid_cells():
    parsed = parse_matrix_csv("Name;Price;Speed\nA;10,5;3\nB;abc;4\n")
    assert parsed.matrix.loc["A", "Price"] == 10.5
    assert np.isnan(parsed.matrix.loc["B", "Price"])
    assert len(parsed.invalid_cells) == 1 and parsed.types is None


def test_csv_errors():
    with pytest.raises(MCDMError, match="two columns"):
        parse_matrix_csv("only\n1\n2\n")
    with pytest.raises(MCDMError, match="Benefit' or 'Cost"):
        parse_matrix_csv("Alt,x\ntype,good\nA,1\n")


def test_template_has_names_and_empty_values():
    parsed = parse_matrix_csv(template_csv(["a", "b"], ["x", "y"], ["Benefit", "Cost"]))
    assert list(parsed.matrix.index) == ["a", "b"]
    assert parsed.types == ["Benefit", "Cost"]
    assert parsed.matrix.isna().all().all()


# ------------------------------------------------------------ explanation & export

def test_explanations_are_consistent(phone_matrix, phone_types):
    for method in (topsis, swm):
        result = method(phone_matrix, phone_types, [0.4, 0.2, 0.3, 0.1])
        table = explain_ranking(result)
        assert table["Alternative"].tolist() == result.ranking["Alternative"].tolist()
        for _, row in table.iterrows():
            strengths = {s.split(" (")[0] for s in row["Main strengths"].split(", ")}
            weaknesses = {s.split(" (")[0] for s in row["Main weaknesses"].split(", ")}
            assert not (strengths & weaknesses - {"—"})


def test_exports_produce_files(phone_matrix, phone_types):
    names = list(phone_matrix.columns)
    weighting = ahp_weights(build_pairwise_matrix(4, {(0, 1): 3, (0, 2): 2}), names)
    ranking = topsis(phone_matrix, phone_types, weighting.weights)
    report = ReportData("Test λ ξ", pd.DataFrame({"Criterion": names, "Type": phone_types,
                                                  "Unit": "", "Description": ""}),
                        phone_matrix, weighting, ranking, explain_ranking(ranking))
    xlsx = excel_report(report)
    assert xlsx[:2] == b"PK"  # zip container
    sheets = pd.read_excel(pd.io.common.BytesIO(xlsx), sheet_name=None)
    assert {"Summary", "Ranking", "Weights", "Decision matrix"} <= set(sheets)
    if pdf_available():
        assert pdf_report(report)[:4] == b"%PDF"
