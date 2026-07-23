from __future__ import annotations

import math
from statistics import NormalDist, mean, stdev
from typing import Iterable


def trial_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = [float(value) for value in values]
    count = len(samples)
    if not samples:
        return {
            "mean": None,
            "sample_sd": None,
            "two_sd": None,
            "ci95_half_width": None,
            "ci95_lower": None,
            "ci95_upper": None,
            "n": 0,
        }
    sample_mean = mean(samples)
    if count < 2:
        return {
            "mean": sample_mean,
            "sample_sd": None,
            "two_sd": None,
            "ci95_half_width": None,
            "ci95_lower": None,
            "ci95_upper": None,
            "n": count,
        }
    sample_sd = stdev(samples)
    half_width = t_critical_95(count - 1) * sample_sd / math.sqrt(count)
    return {
        "mean": sample_mean,
        "sample_sd": sample_sd,
        "two_sd": 2 * sample_sd,
        "ci95_half_width": half_width,
        "ci95_lower": sample_mean - half_width,
        "ci95_upper": sample_mean + half_width,
        "n": count,
    }


_T_CRITICAL_95 = (
    0.0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
    2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093,
    2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045,
    2.042,
)


def t_critical_95(degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive")
    if degrees_of_freedom < len(_T_CRITICAL_95):
        return _T_CRITICAL_95[degrees_of_freedom]
    # Cornish-Fisher expansion for the two-sided 95% Student-t quantile.
    z = NormalDist().inv_cdf(0.975)
    v = float(degrees_of_freedom)
    return (
        z
        + (z**3 + z) / (4 * v)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * v**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * v**3)
    )


PRIMARY_METRICS = {
    "revenue": {
        "label": "Revenue",
        "definition": "Payment collected from the winner, averaged across rounds in each run.",
        "preferred_direction": "higher",
    },
    "allocative_efficiency": {
        "label": "Allocative efficiency",
        "definition": "Winner private value divided by the highest private value, averaged across rounds.",
        "preferred_direction": "higher",
    },
    "payoff": {
        "label": "Payoff",
        "definition": "Winner private value minus payment and zero otherwise, averaged across bidder-rounds.",
        "preferred_direction": "higher",
    },
    "truthfulness_deviation": {
        "label": "Truthfulness deviation",
        "definition": "Absolute bid minus private-value difference, averaged across bidder-rounds.",
        "preferred_direction": "lower",
    },
}
