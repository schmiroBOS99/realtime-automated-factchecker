"""
Dataset loading utilities for evaluation.

Provides smoke test claim sets and utilities to load claims from JSONL files.
"""
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def load_claims_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load claims from a JSONL file.
    
    Expected format: one JSON object per line with at least a "claim" field.
    Optional fields: "expected_label", "category", "notes"
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of claim dictionaries
    """
    claims = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    claim_dict = json.loads(line)
                    if "claim" not in claim_dict:
                        logger.warning(f"Line {line_num}: Missing 'claim' field, skipping")
                        continue
                    claims.append(claim_dict)
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Invalid JSON: {e}, skipping")
                    continue
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading JSONL file {file_path}: {e}")
        return []
    
    logger.info(f"Loaded {len(claims)} claims from {file_path}")
    return claims


def get_smoke_test_claims() -> List[Dict[str, Any]]:
    """
    Get a small smoke test claim set (20-50 claims) covering different categories.
    
    Categories:
    - Person/Office (Präsidenten-/Rollenclaims)
    - Vergleiche (Nil vs Rhein)
    - Numerisch (%/km/Einheiten)
    - Temporal (ist/war/seit)
    - Sport (Jahr/Sieg)
    - NEI/Ambiguität (bewusst unter-spezifiziert)
    
    Returns:
        List of claim dictionaries with "claim" and optional "category" fields
    """
    return [
        # Person/Office
        {"claim": "Frank-Walter Steinmeier ist Präsident Deutschlands.", "category": "person_office"},
        {"claim": "Angela Merkel war Bundeskanzlerin Deutschlands.", "category": "person_office"},
        {"claim": "Emmanuel Macron ist Präsident Frankreichs.", "category": "person_office"},
        
        # Vergleiche
        {"claim": "Der Nil ist länger als der Rhein.", "category": "comparison"},
        {"claim": "Deutschland hat mehr Einwohner als Österreich.", "category": "comparison"},
        
        # Numerisch
        {"claim": "Die CO2-Konzentration in der Atmosphäre beträgt etwa 0,04 Prozent.", "category": "numeric"},
        {"claim": "Der Mount Everest ist 8848 Meter hoch.", "category": "numeric"},
        
        # Temporal
        {"claim": "Angela Merkel ist Bundeskanzlerin Deutschlands.", "category": "temporal"},
        {"claim": "Barack Obama war Präsident der USA.", "category": "temporal"},
        {"claim": "Macron ist seit 2017 Präsident Frankreichs.", "category": "temporal"},
        
        # Sport
        {"claim": "Deutschland gewann die Fußball-WM 2014.", "category": "sport"},
        {"claim": "Bayern München gewann die Champions League 2020.", "category": "sport"},
        {"claim": "Ballack war Kapitän der Deutschen Nationalmannschaft 2006.", "category": "sport"},
        
        # NEI/Ambiguität
        {"claim": "Die Photosynthese findet in Chloroplasten statt.", "category": "general"},
        {"claim": "Der Zweite Weltkrieg endete 1945.", "category": "general"},
        {"claim": "Berlin ist die Hauptstadt Deutschlands.", "category": "general"},
        
        # Edge cases
        {"claim": "Die Sonne ist ein Stern.", "category": "general"},
        {"claim": "Wasser kocht bei 100 Grad Celsius.", "category": "numeric"},
        {"claim": "Die Erde ist rund.", "category": "general"},
    ]


def save_smoke_test_claims(output_path: str = "eval_outputs/smoke_test_claims.jsonl"):
    """
    Save smoke test claims to a JSONL file.
    
    Args:
        output_path: Path to output JSONL file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    claims = get_smoke_test_claims()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for claim_dict in claims:
            f.write(json.dumps(claim_dict, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(claims)} smoke test claims to {output_path}")

