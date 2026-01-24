"""
Unit tests for comparison verifier.
"""
import pytest
from app.pipeline.comparison_verifier import ComparisonVerifier
from app.pipeline.models import Evidence


def test_parse_comparison_claim():
    """Test parsing comparison claims."""
    verifier = ComparisonVerifier()
    
    result = verifier._parse_comparison_claim("Der Nil ist länger als der Rhein")
    assert result is not None
    x_entity, y_entity, comparison_word, attribute = result
    assert "nil" in x_entity.lower()
    assert "rhein" in y_entity.lower()
    assert comparison_word.lower() == "länger"


def test_compare_values():
    """Test numeric value comparison."""
    verifier = ComparisonVerifier()
    
    # Test "länger" (greater than)
    assert verifier._compare_values(100.0, 50.0, "länger") is True
    assert verifier._compare_values(50.0, 100.0, "länger") is False
    
    # Test "kürzer" (less than)
    assert verifier._compare_values(50.0, 100.0, "kürzer") is True
    assert verifier._compare_values(100.0, 50.0, "kürzer") is False


def test_verify_comparison():
    """Test full comparison verification."""
    verifier = ComparisonVerifier()
    
    # Create mock evidences with numeric values
    evidences = [
        Evidence(
            source="wikipedia",
            title="Nil",
            snippet="Der Nil ist 6650 Kilometer lang.",
            url="https://de.wikipedia.org/wiki/Nil"
        ),
        Evidence(
            source="wikipedia",
            title="Rhein",
            snippet="Der Rhein ist 1233 Kilometer lang.",
            url="https://de.wikipedia.org/wiki/Rhein"
        )
    ]
    
    result = verifier.verify(
        claim="Der Nil ist länger als der Rhein",
        evidences=evidences,
        subclaims=["Der Nil ist 6650 Kilometer lang", "Der Rhein ist 1233 Kilometer lang"]
    )
    
    assert result["label"] in ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
