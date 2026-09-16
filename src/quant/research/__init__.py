"""Tools for evaluating predictive factors and their portfolio contribution."""

from .adaptive_weights import WEIGHT_METHODS, AdaptiveFactorWeightEngine, AdaptiveWeightResult
from .alpha import (
    RESEARCH_FACTORS,
    AnnualWalkForwardResearch,
    FactorCombinationResearch,
    FactorNeutralizer,
    ProfessionalFactorEvaluator,
    ProfessionalFactorResearchPipeline,
)
from .attribution import TradeAttributionEngine, TradeAttributionResult
from .candidate import AlphaCandidateRanker, CandidateResult
from .data import FactorResearchDataBuilder
from .decision_explanation import DecisionExplanationEngine
from .error_analysis import ErrorAnalysisResult, PredictionErrorAnalyzer
from .evolution import EvolutionResult, StrategyEvolutionEngine
from .experiments import ExperimentMetadata, ExperimentRegistry
from .factor_evaluation import FactorEvaluator, factor_contribution_analysis
from .leakage import (
    assert_feature_available,
    assert_fundamental_available,
    assert_label_after_signal,
    assert_training_before_test,
)
from .live_simulation import LiveSimulationEngine, LiveSimulationResult
from .optimization import ParameterOptimizer, factor_weight_candidates
from .performance_attribution import PerformanceAttributionEngine, PerformanceAttributionResult
from .preprocessing import ResearchPreprocessor, ResearchPreprocessResult
from .strategy_diagnosis import StrategyDiagnosisEngine, StrategyDiagnosisResult
from .walk_forward import WalkForwardConfig, WalkForwardPeriod, WalkForwardValidator

__all__ = [
    "RESEARCH_FACTORS",
    "WEIGHT_METHODS",
    "AdaptiveFactorWeightEngine",
    "AdaptiveWeightResult",
    "AlphaCandidateRanker",
    "AnnualWalkForwardResearch",
    "CandidateResult",
    "DecisionExplanationEngine",
    "ErrorAnalysisResult",
    "EvolutionResult",
    "ExperimentMetadata",
    "ExperimentRegistry",
    "FactorCombinationResearch",
    "FactorEvaluator",
    "FactorNeutralizer",
    "FactorResearchDataBuilder",
    "LiveSimulationEngine",
    "LiveSimulationResult",
    "ParameterOptimizer",
    "PerformanceAttributionEngine",
    "PerformanceAttributionResult",
    "PredictionErrorAnalyzer",
    "ProfessionalFactorEvaluator",
    "ProfessionalFactorResearchPipeline",
    "ResearchPreprocessResult",
    "ResearchPreprocessor",
    "StrategyDiagnosisEngine",
    "StrategyDiagnosisResult",
    "StrategyEvolutionEngine",
    "TradeAttributionEngine",
    "TradeAttributionResult",
    "WalkForwardConfig",
    "WalkForwardPeriod",
    "WalkForwardValidator",
    "assert_feature_available",
    "assert_fundamental_available",
    "assert_label_after_signal",
    "assert_training_before_test",
    "factor_contribution_analysis",
    "factor_weight_candidates",
]
