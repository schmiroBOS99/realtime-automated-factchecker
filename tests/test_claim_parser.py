"""
Unit tests for claim parser.
"""
import pytest
from app.pipeline.claim_parser import ClaimParser


def test_parse_simple_claim():
    """Test parsing a simple geographic claim."""
    parser = ClaimParser()
    parsed = parser.parse_claim("Berlin ist die Hauptstadt von Deutschland")
    
    assert parsed.subject == "Berlin"
    assert parsed.object == "Deutschland"
    assert parsed.relation_type == "geo_location"
    assert parsed.category == "geography"


def test_parse_comparison_claim():
    """Test parsing a comparison claim."""
    parser = ClaimParser()
    parsed = parser.parse_claim("Der Nil ist länger als der Rhein")
    
    assert parsed.category == "comparison"
    assert parsed.is_hard_claim is True
    assert len(parsed.subclaims) > 0


def test_filter_subjective_claim():
    """Test filtering of subjective claims."""
    parser = ClaimParser()
    
    # Should be filtered
    assert parser.is_personal_or_subjective_claim("Ich finde das gut") is True
    assert parser.is_personal_or_subjective_claim("Was ist das?") is True
    
    # Should not be filtered
    assert parser.is_personal_or_subjective_claim("Berlin ist die Hauptstadt") is False


def test_hard_claim_detection():
    """Test hard claim detection."""
    parser = ClaimParser()
    
    # Comparison claim should be hard
    parsed = parser.parse_claim("Der Nil ist länger als der Rhein")
    assert parsed.is_hard_claim is True
    
    # Simple claim should not be hard
    parsed = parser.parse_claim("Berlin ist die Hauptstadt")
    assert parsed.is_hard_claim is False
