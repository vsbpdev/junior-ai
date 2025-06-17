# Manual Override Controls for Pattern Detection

The Junior AI Assistant includes comprehensive manual override capabilities that give users fine-grained control over pattern detection behavior. This document describes all available manual override features and how to use them.

## Overview

Manual override controls allow you to:
- Enable/disable pattern detection globally or per category
- Add/remove custom keywords for pattern detection
- Add custom regex patterns for advanced matching
- Force AI consultations regardless of pattern detection
- Customize sensitivity settings per category

## Available MCP Tools

### 1. `toggle_pattern_detection`
Enable or disable pattern detection globally.

**Parameters:**
- `enabled` (boolean, required): True to enable, False to disable

**Example:**
```json
{
  "name": "toggle_pattern_detection",
  "arguments": {
    "enabled": false
  }
}
```

### 2. `toggle_category`
Enable or disable a specific pattern category.

**Parameters:**
- `category` (string, required): One of: "security", "uncertainty", "algorithm", "gotcha", "architecture"
- `enabled` (boolean, required): True to enable, False to disable

**Example:**
```json
{
  "name": "toggle_category",
  "arguments": {
    "category": "security",
    "enabled": false
  }
}
```

### 3. `add_pattern_keywords`
Add custom keywords to a pattern category.

**Parameters:**
- `category` (string, required): Pattern category
- `keywords` (array of strings, required): Keywords to add

**Example:**
```json
{
  "name": "add_pattern_keywords",
  "arguments": {
    "category": "security",
    "keywords": ["myApiKey", "customToken", "secretConfig"]
  }
}
```

### 4. `remove_pattern_keywords`
Remove keywords from a pattern category.

**Parameters:**
- `category` (string, required): Pattern category
- `keywords` (array of strings, required): Keywords to remove

**Example:**
```json
{
  "name": "remove_pattern_keywords",
  "arguments": {
    "category": "uncertainty",
    "keywords": ["maybe", "perhaps"]
  }
}
```

### 5. `list_pattern_keywords`
List all keywords for a pattern category.

**Parameters:**
- `category` (string, required): Pattern category

**Example:**
```json
{
  "name": "list_pattern_keywords",
  "arguments": {
    "category": "algorithm"
  }
}
```

### 6. `force_consultation`
Force AI consultation regardless of pattern detection.

**Parameters:**
- `context` (string, required): Text/code to analyze
- `category` (string, required): Pattern category to use
- `multi_ai` (boolean, optional): Use multiple AIs (default: false)

**Example:**
```json
{
  "name": "force_consultation",
  "arguments": {
    "context": "def calculate_hash(data):\n    return data[:10]",
    "category": "security",
    "multi_ai": true
  }
}
```

## Configuration Schema

The manual override configuration is stored in the `credentials.json` file:

```json
{
  "pattern_detection": {
    "enabled": true,
    "manual_override": {
      "allow_disable_detection": true,
      "allow_force_consultation": true,
      "allow_custom_patterns": true,
      "force_consultation_keywords": [],
      "global_exclusions": [],
      "bypass_patterns": []
    },
    "pattern_categories": {
      "security": {
        "enabled": true,
        "custom_keywords": [],
        "custom_patterns": [],
        "disabled_keywords": []
      }
      // ... other categories
    }
  }
}
```

## Use Cases

### 1. Reducing False Positives
If certain keywords are triggering unnecessary consultations:

```bash
# Remove overly generic keywords
mcp__junior-ai__remove_pattern_keywords category="uncertainty" keywords='["maybe", "possibly"]'
```

### 2. Domain-Specific Keywords
Add keywords specific to your project:

```bash
# Add project-specific security keywords
mcp__junior-ai__add_pattern_keywords category="security" keywords='["apiSecret", "dbPassword", "authToken"]'
```

### 3. Temporary Disable
Disable pattern detection during bulk operations:

```bash
# Disable globally
mcp__junior-ai__toggle_pattern_detection enabled=false

# Or disable specific category
mcp__junior-ai__toggle_category category="algorithm" enabled=false
```

### 4. Manual Security Review
Force a security consultation for code that might not trigger patterns:

```bash
mcp__junior-ai__force_consultation context="function processPayment(amount, card) { ... }" category="security" multi_ai=true
```

## Advanced Usage

### Custom Regex Patterns
You can add custom regex patterns by modifying the configuration directly:

```json
{
  "pattern_categories": {
    "security": {
      "custom_patterns": [
        "\\bapi[_-]?v\\d+\\b",  // API version patterns
        "\\b[A-Z0-9]{32}\\b"     // 32-character hex strings
      ]
    }
  }
}
```

### Sensitivity Overrides
Combine manual overrides with sensitivity settings:

```bash
# Set high sensitivity for security, but disable specific keywords
mcp__junior-ai__update_sensitivity category_overrides='{"security": "high"}'
mcp__junior-ai__remove_pattern_keywords category="security" keywords='["key", "token"]'
```

## Best Practices

1. **Test Before Disabling**: Before disabling a category or removing keywords, test with sample code to ensure you're not missing important patterns.

2. **Document Changes**: Keep track of custom keywords and disabled categories in your project documentation.

3. **Regular Review**: Periodically review your manual overrides to ensure they're still relevant.

4. **Use Force Consultation Sparingly**: While useful, forced consultations bypass optimization and should be used judiciously.

5. **Category-Specific Approach**: Rather than disabling globally, consider disabling specific categories that aren't relevant to your project.

## Troubleshooting

### Changes Not Taking Effect
- Ensure the configuration file has proper write permissions
- Check that the JSON syntax is valid after manual edits
- Restart the MCP server if making direct file edits

### Keywords Not Matching
- Keywords are case-insensitive
- Ensure keywords are whole words (not substrings)
- Check for typos in category names

### Performance Impact
- Adding many custom regex patterns may impact performance
- Consider using simple keywords instead of complex patterns when possible

## Integration with Other Features

Manual overrides work seamlessly with:
- **Sensitivity Levels**: Override settings apply regardless of sensitivity
- **Async Caching**: Custom patterns are cached for performance
- **AI Consultation Manager**: Force consultation uses the same AI selection logic
- **Context-Aware Matching**: Custom keywords benefit from context analysis

## Examples

### Example 1: Project-Specific Configuration
```bash
# Disable architecture patterns (not needed for small project)
mcp__junior-ai__toggle_category category="architecture" enabled=false

# Add project-specific security terms
mcp__junior-ai__add_pattern_keywords category="security" keywords='["userSecret", "appKey", "serviceToken"]'

# Remove generic terms that cause false positives
mcp__junior-ai__remove_pattern_keywords category="uncertainty" keywords='["maybe", "might"]'
```

### Example 2: Security-Focused Setup
```bash
# Maximum security sensitivity
mcp__junior-ai__update_sensitivity global_level="high" category_overrides='{"security": "maximum"}'

# Add compliance-specific keywords
mcp__junior-ai__add_pattern_keywords category="security" keywords='["pii", "gdpr", "hipaa", "pci"]'

# Force multi-AI consultation for any security patterns
# (This would be done through configuration)
```

### Example 3: Performance Optimization
```bash
# Disable less critical categories
mcp__junior-ai__toggle_category category="gotcha" enabled=false
mcp__junior-ai__toggle_category category="architecture" enabled=false

# Keep only essential security and algorithm checks
mcp__junior-ai__update_sensitivity global_level="low"
```

This documentation provides comprehensive coverage of the manual override functionality, making it easy for users to customize pattern detection behavior to their specific needs.