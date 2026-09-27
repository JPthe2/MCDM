"""Criteria weighting and alternative ranking methods, implemented with NumPy/SciPy."""

from methods.ahp import ahp_weights, build_pairwise_matrix
from methods.bwm import bwm_weights
from methods.critic import critic_weights
from methods.entropy import entropy_weights
from methods.equal import equal_weights
from methods.swm import swm
from methods.topsis import topsis

WEIGHTING_METHODS = ("AHP", "BWM", "Entropy", "CRITIC", "Equal weights")
RANKING_METHODS = ("TOPSIS", "SWM")

__all__ = [
    "RANKING_METHODS", "WEIGHTING_METHODS", "ahp_weights", "build_pairwise_matrix",
    "bwm_weights", "critic_weights", "entropy_weights", "equal_weights", "swm", "topsis",
]
