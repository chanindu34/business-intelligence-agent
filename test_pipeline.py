from app import calculate, CalculateInput, SearchKBInput
from pydantic import ValidationError
import pytest

def test_calculate_basic_multiplication():
    assert calculate("47 * 12") == "564"

def test_calculate_division():
    assert calculate("10 / 4") == "2.5"

def test_calculate_division_by_zero():
    assert calculate("10 / 0") == "Error: Division by zero."

def test_calculate_invalid_syntax():
    result = calculate("5 +")
    assert result.startswith("Error:")

def test_calculate_input_validation_missing_field():
    with pytest.raises(ValidationError):
        CalculateInput()

def test_calculate_input_validation_wrong_type():
    with pytest.raises(ValidationError):
        CalculateInput(expression=47)

def test_search_kb_input_validation():
    with pytest.raises(ValidationError):
        SearchKBInput()

def test_calculate_input_valid():
    validated = CalculateInput(expression="2 + 2")
    assert validated.expression == "2 + 2"
    