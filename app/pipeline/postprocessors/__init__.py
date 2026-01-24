"""
PostProcessor modules for Verifier.

PostProcessors adjust NLI kernel decisions based on domain-specific heuristics.
Each processor is optional and can be enabled/disabled independently.
"""
from app.pipeline.postprocessors.base import PostProcessor, Decision

# Import processors (lazy to avoid circular imports)
def get_all_processors():
    """Get all available postprocessors."""
    from app.pipeline.postprocessors.temporal import TemporalPostProcessor
    from app.pipeline.postprocessors.yearmatch import YearMatchPostProcessor
    from app.pipeline.postprocessors.negation import NegationPostProcessor
    from app.pipeline.postprocessors.predicate_guard import PredicateGuardPostProcessor
    from app.pipeline.postprocessors.dynamic_threshold import DynamicThresholdPostProcessor
    from app.pipeline.postprocessors.sports_year import SportsYearPostProcessor
    from app.pipeline.postprocessors.subject_voting import SubjectVotingPostProcessor
    from app.pipeline.postprocessors.population_verifier import PopulationVerifierPostProcessor
    from app.pipeline.postprocessors.geo_verifier import GeoVerifierPostProcessor
    
    return [
        SubjectVotingPostProcessor(),  # Early decision processor (should run first)
        TemporalPostProcessor(),
        YearMatchPostProcessor(),
        NegationPostProcessor(),
        PopulationVerifierPostProcessor(),  # Population-specific verification (before dynamic_threshold)
        GeoVerifierPostProcessor(),  # Geo location verification (before dynamic_threshold)
        DynamicThresholdPostProcessor(),
        PredicateGuardPostProcessor(),
        SportsYearPostProcessor(),
    ]

__all__ = [
    "PostProcessor",
    "Decision",
    "get_all_processors",
]

