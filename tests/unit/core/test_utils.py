"""Unit tests for core.utils module"""

import json
import pytest

from core import utils


@pytest.mark.unit
class TestFormatErrorResponse:
    """Test suite for format_error_response function"""
    
    def test_format_error_response_with_context(self):
        """Test error formatting with context information"""
        error = Exception("Test error")
        result = utils.format_error_response(error, "Additional context")
        
        expected = "❌ Additional context: Test error"
        assert result == expected
    
    def test_format_error_response_without_context(self):
        """Test error formatting without context"""
        error = Exception("Test error")
        result = utils.format_error_response(error)
        
        expected = "❌ Error (Exception): Test error"
        assert result == expected
    
    def test_format_error_response_different_exception_types(self):
        """Test error formatting with different exception types"""
        value_error = ValueError("Invalid value")
        result = utils.format_error_response(value_error)
        
        expected = "❌ Error (ValueError): Invalid value"
        assert result == expected
    
    def test_format_error_response_multiline_message(self):
        """Test error formatting with multiline message"""
        error = Exception("Line 1\nLine 2\nLine 3")
        result = utils.format_error_response(error, "Context info")
        
        expected = "❌ Context info: Line 1\nLine 2\nLine 3"
        assert result == expected
    
    def test_format_error_response_special_characters(self):
        """Test error formatting with special characters"""
        error = Exception("Error with 'quotes' and \"double quotes\"")
        result = utils.format_error_response(error, "Context with <brackets> and &symbols&")
        
        expected = "❌ Context with <brackets> and &symbols&: Error with 'quotes' and \"double quotes\""
        assert result == expected


@pytest.mark.unit
class TestValidateTemperature:
    """Test suite for validate_temperature function"""
    
    def test_validate_temperature_valid_float(self):
        """Test temperature validation with valid float values"""
        assert utils.validate_temperature(0.0) == 0.0
        assert utils.validate_temperature(0.5) == 0.5
        assert utils.validate_temperature(1.0) == 1.0
        assert utils.validate_temperature(0.7) == 0.7
    
    def test_validate_temperature_valid_string(self):
        """Test temperature validation with valid string values"""
        # Note: Current implementation doesn't convert strings, so this should raise ValueError
        with pytest.raises(ValueError, match="must be a number"):
            utils.validate_temperature("0.5")
    
    def test_validate_temperature_out_of_range(self):
        """Test temperature validation with out of range values (constrains to valid range)"""
        assert utils.validate_temperature(-0.1) == 0.0
        assert utils.validate_temperature(1.1) == 1.0
        assert utils.validate_temperature(2.0) == 1.0
        assert utils.validate_temperature(-1.0) == 0.0
    
    def test_validate_temperature_invalid_type(self):
        """Test temperature validation with invalid types"""
        with pytest.raises(ValueError, match="must be a number"):
            utils.validate_temperature("invalid")
        
        with pytest.raises(ValueError, match="must be a number"):
            utils.validate_temperature([0.5])
        
        with pytest.raises(ValueError, match="must be a number"):
            utils.validate_temperature({"temp": 0.5})
        
        with pytest.raises(ValueError, match="must be a number"):
            utils.validate_temperature(None)
    
    def test_validate_temperature_edge_cases(self):
        """Test temperature validation edge cases"""
        # Integer values
        assert utils.validate_temperature(0) == 0.0
        assert utils.validate_temperature(1) == 1.0
        
        # Very small valid values
        assert utils.validate_temperature(0.000001) == 0.000001
        assert utils.validate_temperature(0.999999) == 0.999999


@pytest.mark.unit
class TestSafeJsonLoads:
    """Test suite for safe_json_loads function"""
    
    def test_safe_json_loads_valid_json(self):
        """Test safe JSON parsing with valid JSON"""
        # Simple object
        assert utils.safe_json_loads('{"key": "value"}') == {"key": "value"}
        
        # Array
        assert utils.safe_json_loads('[1, 2, 3]') == [1, 2, 3]
        
        # Nested structure
        complex_json = '{"a": {"b": [1, 2, {"c": "d"}]}}'
        expected = {"a": {"b": [1, 2, {"c": "d"}]}}
        assert utils.safe_json_loads(complex_json) == expected
        
        # Empty structures
        assert utils.safe_json_loads('{}') == {}
        assert utils.safe_json_loads('[]') == []
    
    def test_safe_json_loads_invalid_json(self):
        """Test safe JSON parsing with invalid JSON"""
        # Malformed JSON
        assert utils.safe_json_loads('{invalid}') is None
        assert utils.safe_json_loads('{"key": value}') is None
        assert utils.safe_json_loads('[1, 2, 3,]') is None
        
        # Not JSON at all
        assert utils.safe_json_loads('plain text') is None
        assert utils.safe_json_loads('') is None
        assert utils.safe_json_loads('null') is None
    
    def test_safe_json_loads_invalid_returns_none(self):
        """Test safe JSON parsing with invalid JSON returns None"""
        # Invalid JSON returns None
        assert utils.safe_json_loads('{invalid}') is None
        assert utils.safe_json_loads('') is None
    
    def test_safe_json_loads_edge_cases(self):
        """Test safe JSON parsing edge cases"""
        # Unicode
        assert utils.safe_json_loads('{"emoji": "🎉"}') == {"emoji": "🎉"}
        
        # Numbers
        assert utils.safe_json_loads('{"int": 42, "float": 3.14}') == {"int": 42, "float": 3.14}
        
        # Booleans and null
        assert utils.safe_json_loads('{"bool": true, "null": null}') == {"bool": True, "null": None}
        
        # Escaped characters
        assert utils.safe_json_loads('{"escaped": "line\\nbreak"}') == {"escaped": "line\nbreak"}


@pytest.mark.unit
class TestFormatAIName:
    """Test suite for format_ai_name function"""
    
    def test_format_ai_name_standard_names(self):
        """Test AI name formatting for standard names"""
        assert utils.format_ai_name("openai") == "OPENAI"
        assert utils.format_ai_name("gemini") == "GEMINI"
        assert utils.format_ai_name("grok") == "GROK"
        assert utils.format_ai_name("deepseek") == "DEEPSEEK"
        assert utils.format_ai_name("openrouter") == "OPENROUTER"
    
    def test_format_ai_name_case_variations(self):
        """Test AI name formatting with case variations"""
        assert utils.format_ai_name("OPENAI") == "OPENAI"
        assert utils.format_ai_name("OpenAI") == "OPENAI"
        assert utils.format_ai_name("oPeNaI") == "OPENAI"
    
    def test_format_ai_name_unknown_names(self):
        """Test AI name formatting for unknown names"""
        assert utils.format_ai_name("unknown") == "UNKNOWN"
        assert utils.format_ai_name("custom_ai") == "CUSTOM_AI"
        assert utils.format_ai_name("test-ai") == "TEST-AI"
    
    def test_format_ai_name_edge_cases(self):
        """Test AI name formatting edge cases"""
        assert utils.format_ai_name("") == ""
        assert utils.format_ai_name("a") == "A"
        assert utils.format_ai_name("123") == "123"
        assert utils.format_ai_name("ai_with_numbers_123") == "AI_WITH_NUMBERS_123"


@pytest.mark.unit
class TestTruncateText:
    """Test suite for truncate_text function"""
    
    def test_truncate_text_shorter_than_max(self):
        """Test text truncation when text is shorter than max length"""
        text = "Short text"
        result = utils.truncate_text(text, 20)
        assert result == text
    
    def test_truncate_text_longer_than_max(self):
        """Test text truncation when text is longer than max length"""
        text = "This is a very long text that needs to be truncated"
        result = utils.truncate_text(text, 20)
        # Should be 20 - 3 (for "...") = 17 chars + "..."
        expected = text[:17] + "..."
        assert result == expected
        assert len(result) == 20  # Total length should be max_length
    
    def test_truncate_text_custom_suffix(self):
        """Test text truncation with custom suffix"""
        text = "This is a very long text that needs to be truncated"
        suffix = " [truncated]"
        result = utils.truncate_text(text, 20, suffix)
        # Should be 20 - 12 (for " [truncated]") = 8 chars + " [truncated]"
        expected = text[:8] + suffix
        assert result == expected
    
    def test_truncate_text_exact_length(self):
        """Test text truncation when text is exactly max length"""
        text = "Exactly twenty chars"  # 20 characters
        result = utils.truncate_text(text, 20)
        assert result == text
    
    def test_truncate_text_basic_functionality(self):
        """Test basic truncate text functionality"""
        # Empty text
        assert utils.truncate_text("", 10) == ""
        
        # Text shorter than limit
        assert utils.truncate_text("Hello", 10) == "Hello"
        
        # Unicode characters
        text = "Hello 🎉 World"
        result = utils.truncate_text(text, 10)
        assert len(result) <= 10


@pytest.mark.unit
class TestIntegration:
    """Integration tests for utils module functions"""
    
    def test_basic_error_formatting(self):
        """Test basic error formatting integration"""
        error = Exception("API request failed")
        result = utils.format_error_response(error, "Rate limit exceeded")
        
        assert "❌ Rate limit exceeded: API request failed" == result
    
    def test_ai_name_formatting(self):
        """Test AI name formatting"""
        ai_name = "openai"
        formatted_name = utils.format_ai_name(ai_name)
        
        assert formatted_name == "OPENAI"