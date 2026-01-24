#!/usr/bin/env python3
"""
CLI runner for evaluation harness.

Supports running evaluation runs with different presets/feature flags.
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.factchecker import FactChecker
from app.pipeline.eval.harness import EvaluationHarness
from app.pipeline.eval.datasets import load_claims_from_jsonl, get_smoke_test_claims, save_smoke_test_claims

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


PRESETS = {
    "baseline_stage1": {
        "enable_crossencoder_rerank": False,
        "enable_postprocessors": True,
        "enable_decomposition": False,
        "enable_claim_cache": False,
    },
    "crossencoder_on": {
        "enable_crossencoder_rerank": True,
        "enable_postprocessors": True,
        "enable_decomposition": False,
        "enable_claim_cache": False,
    },
    "kernel_only": {
        "enable_crossencoder_rerank": False,
        "enable_postprocessors": False,
        "enable_decomposition": False,
        "enable_claim_cache": False,
    },
    "decomposition_on": {
        "enable_crossencoder_rerank": False,
        "enable_postprocessors": True,
        "enable_decomposition": True,
        "enable_claim_cache": False,
    },
    "full_2_0": {
        "enable_crossencoder_rerank": True,
        "enable_postprocessors": True,
        "enable_decomposition": True,
        "enable_claim_cache": True,
    },
}


def create_factchecker_with_preset(preset_config: Dict[str, Any]) -> FactChecker:
    """
    Create FactChecker instance with preset configuration.
    
    Note: Some flags (crossencoder, postprocessors) require changes to FactChecker internals.
    For now, we use the defaults and log the intended configuration.
    """
    # TODO: FactChecker needs to support these flags in __init__ or start()
    # For now, we create with defaults and log the preset
    factchecker = FactChecker(
        enable_claim_cache=preset_config.get("enable_claim_cache", False),
    )
    
    logger.info(f"PRESET_CONFIG: {preset_config}")
    logger.warning(
        "Note: enable_crossencoder_rerank and enable_postprocessors require "
        "FactChecker modifications (not yet implemented in this version)"
    )
    
    return factchecker


async def main():
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()) + ["all"],
        default="baseline_stage1",
        help="Preset configuration to use",
    )
    parser.add_argument(
        "--claims-file",
        type=str,
        default=None,
        help="Path to JSONL file with claims (default: use smoke test claims)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save-smoke-test",
        action="store_true",
        help="Save smoke test claims to JSONL file and exit",
    )
    
    args = parser.parse_args()
    
    # Save smoke test claims if requested
    if args.save_smoke_test:
        smoke_test_path = Path(args.output_dir) / "smoke_test_claims.jsonl"
        save_smoke_test_claims(str(smoke_test_path))
        logger.info(f"Saved smoke test claims to {smoke_test_path}")
        return
    
    # Load claims
    if args.claims_file:
        claims = load_claims_from_jsonl(args.claims_file)
    else:
        claims = get_smoke_test_claims()
        logger.info(f"Using smoke test claims ({len(claims)} claims)")
    
    if not claims:
        logger.error("No claims to evaluate")
        return
    
    # Determine presets to run
    if args.preset == "all":
        presets_to_run = PRESETS.keys()
    else:
        presets_to_run = [args.preset]
    
    # Run evaluation for each preset
    results = {}
    for preset_name in presets_to_run:
        logger.info(f"Running preset: {preset_name}")
        
        preset_config = PRESETS[preset_name]
        factchecker = create_factchecker_with_preset(preset_config)
        
        try:
            await factchecker.start()
            
            harness = EvaluationHarness(
                factchecker=factchecker,
                output_dir=args.output_dir,
            )
            
            jsonl_path, summary_path = await harness.run_evaluation(
                claims=claims,
                preset_name=preset_name,
            )
            
            logger.info(f"Preset '{preset_name}' complete:")
            logger.info(f"  Results: {jsonl_path}")
            logger.info(f"  Summary: {summary_path}")
            
            # Load summary for display
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            print(f"\n=== Summary for '{preset_name}' ===")
            print(f"Total claims: {summary['total_claims']}")
            print(f"Latency (ms): mean={summary['latency_stats']['total_ms']['mean']:.1f}, "
                  f"p50={summary['latency_stats']['total_ms']['p50']:.1f}, "
                  f"p95={summary['latency_stats']['total_ms']['p95']:.1f}")
            print(f"Label distribution: {summary['label_distribution']}")
            print(f"Cache hit rate: {summary['usage_rates']['cache_hit_rate']:.1f}%")
            print(f"Decomposition rate: {summary['usage_rates']['decomposition_rate']:.1f}%")
            
        except Exception as e:
            logger.error(f"Error running preset '{preset_name}': {e}", exc_info=True)
        finally:
            await factchecker.close()
    
    logger.info("Evaluation complete")


if __name__ == "__main__":
    asyncio.run(main())

