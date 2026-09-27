"""Read and write decision matrices as CSV.

Format: the first column holds alternative names and the header row holds criterion
names. An optional row whose first cell is ``type`` gives each criterion's type
(``Benefit`` or ``Cost``)::

    Alternative,Price,Quality
    type,Cost,Benefit
    Supplier A,42,7
    Supplier B,38,6
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.models import MCDMError
from core.validation import validate_names

TYPE_ROW_LABELS = {"type", "types", "criterion type", "__type__"}


@dataclass
class ParsedMatrix:
    """A decision matrix read from CSV, possibly with invalid cells set to NaN."""

    matrix: pd.DataFrame
    types: list[str] | None
    invalid_cells: list[str] = field(default_factory=list)


def parse_matrix_csv(source: bytes | str | io.IOBase) -> ParsedMatrix:
    """Parse an uploaded CSV into a decision matrix.

    Non-numeric cells become NaN and are listed in ``invalid_cells`` so the user can
    fix them in the editor (nothing is silently dropped).

    Raises:
        MCDMError: if the file cannot be parsed or has no criteria/alternatives.
    """
    if isinstance(source, bytes):
        text = source.decode("utf-8-sig")
    elif isinstance(source, str):
        text = source
    else:
        text = source.read()
        text = text.decode("utf-8-sig") if isinstance(text, bytes) else text
    header = text.lstrip().splitlines()[0] if text.strip() else ""
    separator = max([",", ";", "\t"], key=header.count)
    try:
        raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False,
                          skipinitialspace=True, sep=separator)
    except Exception as exc:  # pandas raises several parser error types
        raise MCDMError(f"Could not read the CSV file: {exc}") from exc
    if raw.shape[1] < 2:
        raise MCDMError("The CSV needs at least two columns: alternative names followed by "
                        "one column per criterion.")
    raw = raw.set_index(raw.columns[0])
    raw.index = raw.index.map(lambda s: str(s).strip())
    raw.columns = [str(c).strip() for c in raw.columns]

    types: list[str] | None = None
    type_rows = [label for label in raw.index if label.lower() in TYPE_ROW_LABELS]
    if type_rows:
        type_values = raw.loc[type_rows[0]]
        if isinstance(type_values, pd.DataFrame):
            type_values = type_values.iloc[0]
        types = []
        for criterion, value in type_values.items():
            value = str(value).strip().capitalize()
            if value not in ("Benefit", "Cost"):
                raise MCDMError(f"Type of criterion '{criterion}' must be 'Benefit' or "
                                f"'Cost', found '{value}'.")
            types.append(value)
        raw = raw.drop(index=type_rows)

    if raw.shape[0] == 0:
        raise MCDMError("The CSV contains no alternatives (data rows).")
    validate_names(list(raw.index), "alternative")
    validate_names(list(raw.columns), "criterion")

    numeric = raw.apply(lambda col: pd.to_numeric(col.str.replace(",", ".", regex=False)
                                                  .str.strip(), errors="coerce"))
    invalid = [f"{raw.index[i]} / {raw.columns[j]} ('{raw.iat[i, j]}')"
               for i, j in zip(*np.nonzero(numeric.isna().to_numpy()))]
    return ParsedMatrix(numeric.astype(float), types, invalid)


def matrix_to_csv(matrix: pd.DataFrame, types: Sequence[str] | None = None) -> bytes:
    """Serialise a matrix (and optionally a type row) in the format read above."""
    frame = matrix.copy()
    frame.index = frame.index.map(str)
    if types is not None:
        type_row = pd.DataFrame([list(types)], columns=frame.columns, index=["type"])
        frame = pd.concat([type_row, frame.astype(object)])
    frame.index.name = "Alternative"
    return frame.to_csv().encode("utf-8-sig")


def template_csv(alternatives: Sequence[str], criteria: Sequence[str],
                 types: Sequence[str]) -> bytes:
    """An empty matrix template with the current names and types."""
    empty = pd.DataFrame("", index=list(alternatives), columns=list(criteria))
    return matrix_to_csv(empty, types)
