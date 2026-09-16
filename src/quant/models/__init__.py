"""Model factories and time-ordered ranking research models."""

from .alpha_ensemble import AlphaEnsemble, AlphaEnsembleWeights
from .ranking import (
    RankingLabelEncoding,
    RankingModel,
    TimeSplit,
    add_future_excess_return,
    encode_ranking_labels,
    prediction_ic_metrics,
    time_ordered_split,
    validate_ranking_labels,
)

__all__ = [
    "AlphaEnsemble",
    "AlphaEnsembleWeights",
    "RankingLabelEncoding",
    "RankingModel",
    "TimeSplit",
    "add_future_excess_return",
    "encode_ranking_labels",
    "prediction_ic_metrics",
    "time_ordered_split",
    "validate_ranking_labels",
]
