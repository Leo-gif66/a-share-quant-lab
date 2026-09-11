"""Tools for evaluating predictive factors and their portfolio contribution."""

from .data import FactorResearchDataBuilder
from .factor_evaluation import FactorEvaluator, factor_contribution_analysis
from .optimization import ParameterOptimizer, factor_weight_candidates
from .walk_forward import WalkForwardConfig, WalkForwardPeriod, WalkForwardValidator
from .preprocessing import ResearchPreprocessResult, ResearchPreprocessor
from .adaptive_weights import AdaptiveFactorWeightEngine, AdaptiveWeightResult, WEIGHT_METHODS
from .attribution import TradeAttributionEngine, TradeAttributionResult
from .decision_explanation import DecisionExplanationEngine
from .error_analysis import ErrorAnalysisResult, PredictionErrorAnalyzer
from .live_simulation import LiveSimulationEngine, LiveSimulationResult
from .strategy_diagnosis import StrategyDiagnosisEngine, StrategyDiagnosisResult
from .candidate import AlphaCandidateRanker, CandidateResult
from .evolution import EvolutionResult, StrategyEvolutionEngine
from .performance_attribution import PerformanceAttributionEngine, PerformanceAttributionResult
from .alpha import (
    AnnualWalkForwardResearch,
    FactorCombinationResearch,
    FactorNeutralizer,
    ProfessionalFactorEvaluator,
    ProfessionalFactorResearchPipeline,
    RESEARCH_FACTORS,
)

__all__ = [
    "FactorEvaluator",
    "FactorCombinationResearch",
    "FactorNeutralizer",
    "FactorResearchDataBuilder",
    "ParameterOptimizer",
    "ProfessionalFactorEvaluator",
    "ProfessionalFactorResearchPipeline",
    "RESEARCH_FACTORS",
    "ResearchPreprocessResult",
    "ResearchPreprocessor",
    "AnnualWalkForwardResearch",
    "AdaptiveFactorWeightEngine",
    "AlphaCandidateRanker",
    "AdaptiveWeightResult",
    "DecisionExplanationEngine",
    "ErrorAnalysisResult",
    "CandidateResult",
    "EvolutionResult",
    "LiveSimulationEngine",
    "LiveSimulationResult",
    "PredictionErrorAnalyzer",
    "PerformanceAttributionEngine",
    "PerformanceAttributionResult",
    "StrategyDiagnosisEngine",
    "StrategyDiagnosisResult",
    "StrategyEvolutionEngine",
    "TradeAttributionEngine",
    "TradeAttributionResult",
    "WEIGHT_METHODS",
    "WalkForwardConfig",
    "WalkForwardPeriod",
    "WalkForwardValidator",
    "factor_contribution_analysis",
    "factor_weight_candidates",
]
