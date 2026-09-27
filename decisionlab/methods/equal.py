"""Equal weights — the neutral baseline (w_j = 1/n)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from core.models import WeightingResult
from core.validation import validate_names


def equal_weights(criteria: Sequence[str]) -> WeightingResult:
    """Give every criterion the same weight ``1/n``."""
    names = validate_names(criteria, "criterion")
    n = len(names)
    weights = pd.Series(np.full(n, 1.0 / n), index=pd.Index(names), name="Weight")
    return WeightingResult("Equal weights", weights, {}, [])
