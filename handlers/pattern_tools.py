"""Handlers for pattern detection tool calls.

This module implements comprehensive pattern detection and AI consultation
tools. It provides interfaces for detecting code patterns that may benefit
from AI analysis, managing pattern detection sensitivity, and controlling
the AI consultation process.

Pattern detection tools:
- pattern_check: Analyze text for patterns requiring AI consultation
- junior_consult: Smart AI consultation based on detected patterns
- pattern_stats: View pattern detection statistics
- get_sensitivity_config: View current sensitivity settings
- update_sensitivity: Modify detection sensitivity levels

Manual override tools:
- toggle_pattern_detection: Enable/disable pattern detection globally
- toggle_category: Enable/disable specific pattern categories
- add_pattern_keywords: Add custom keywords to categories
- remove_pattern_keywords: Remove keywords from categories
- list_pattern_keywords: List all keywords for a category
- force_consultation: Force AI consultation regardless of patterns

AI consultation manager tools:
- ai_consultation_strategy: Get recommended AI strategy for patterns
- ai_consultation_metrics: View AI consultation performance metrics
- ai_consultation_audit: Access consultation history and audit trail
- ai_governance_report: Export compliance and governance reports
"""

from typing import Dict, Any, List
from .base import BaseHandler


class PatternToolsHandler(BaseHandler):
    """Handles pattern detection related tool calls."""
    
    def get_tool_names(self) -> List[str]:
        """Return list of tool names this handler supports."""
        return [
            "pattern_check",
            "junior_consult",
            "pattern_stats",
            "get_sensitivity_config",
            "update_sensitivity",
            "toggle_pattern_detection",
            "toggle_category",
            "add_pattern_keywords",
            "remove_pattern_keywords",
            "list_pattern_keywords",
            "force_consultation",
            # AI consultation manager tools
            "ai_consultation_strategy",
            "ai_consultation_metrics",
            "ai_consultation_audit",
            "ai_governance_report"
        ]
    
    def handle(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Handle a pattern detection tool call."""
        if tool_name == "pattern_check":
            return self._handle_pattern_check(arguments)
        elif tool_name == "junior_consult":
            return self._handle_junior_consult(arguments)
        elif tool_name == "pattern_stats":
            return self._handle_pattern_stats(arguments)
        elif tool_name == "get_sensitivity_config":
            return self._handle_get_sensitivity_config(arguments)
        elif tool_name == "update_sensitivity":
            return self._handle_update_sensitivity(arguments)
        elif tool_name == "toggle_pattern_detection":
            return self._handle_toggle_pattern_detection(arguments)
        elif tool_name == "toggle_category":
            return self._handle_toggle_category(arguments)
        elif tool_name == "add_pattern_keywords":
            return self._handle_add_pattern_keywords(arguments)
        elif tool_name == "remove_pattern_keywords":
            return self._handle_remove_pattern_keywords(arguments)
        elif tool_name == "list_pattern_keywords":
            return self._handle_list_pattern_keywords(arguments)
        elif tool_name == "force_consultation":
            return self._handle_force_consultation(arguments)
        elif tool_name == "ai_consultation_strategy":
            return self._handle_ai_consultation_strategy(arguments)
        elif tool_name == "ai_consultation_metrics":
            return self._handle_ai_consultation_metrics(arguments)
        elif tool_name == "ai_consultation_audit":
            return self._handle_ai_consultation_audit(arguments)
        elif tool_name == "ai_governance_report":
            return self._handle_ai_governance_report(arguments)
        
        return f"❌ Unknown pattern tool: {tool_name}"
    
    def _handle_pattern_check(self, arguments: Dict[str, Any]) -> str:
        """Handle pattern check request."""
        text = arguments.get('text', '')
        auto_consult = arguments.get('auto_consult', True)
        
        if not text:
            return "❌ Missing required parameter: text"
        
        if not self.pattern_engine:
            return "❌ Pattern detection not available"
        
        try:
            # Detect patterns
            patterns = self.pattern_engine.detect_patterns(text)
            
            if not patterns:
                return "✅ No significant patterns detected"
            
            # Format results
            output = "# Pattern Detection Results\n\n"
            
            for pattern in patterns:
                output += f"## {pattern['category'].title()} Pattern Detected\n"
                output += f"- **Severity**: {pattern['severity']}\n"
                output += f"- **Context**: {pattern['context'][:200]}...\n"
                output += f"- **Line**: {pattern.get('line_number', 'N/A')}\n\n"
            
            # Auto-consult if requested and patterns found
            if auto_consult and self.response_manager:
                consultation = self.response_manager.get_response(patterns, text)
                if consultation:
                    output += "\n## AI Consultation\n\n"
                    output += consultation
            
            return output
            
        except Exception as e:
            return f"❌ Error during pattern detection: {str(e)}"
    
    def _handle_junior_consult(self, arguments: Dict[str, Any]) -> str:
        """Handle smart AI consultation request."""
        context = arguments.get('context', '')
        force_multi_ai = arguments.get('force_multi_ai', False)
        
        if not context:
            return "❌ Missing required parameter: context"
        
        if not self.response_manager:
            return "❌ AI consultation not available"
        
        try:
            # First detect patterns
            patterns = []
            if self.pattern_engine:
                patterns = self.pattern_engine.detect_patterns(context)
            
            # Get consultation
            if patterns or force_multi_ai:
                response = self.response_manager.get_response(
                    patterns if patterns else [],
                    context,
                    force_multi_ai=force_multi_ai
                )
                return response if response else "❌ No consultation response generated"
            
            # No patterns detected, use default AI
            from ai.caller import call_ai
            default_ai = self.credentials.get('pattern_detection', {}).get('default_junior', 'openrouter')
            if default_ai in self.ai_clients:
                return call_ai(default_ai, context)
            
            return "❌ Default AI not available and no patterns detected"
                    
        except Exception as e:
            return f"❌ Error during consultation: {str(e)}"
    
    def _handle_pattern_stats(self, arguments: Dict[str, Any]) -> str:
        """Get pattern detection statistics."""
        if not self.pattern_engine:
            return "❌ Pattern detection not available"
        
        try:
            stats = self.pattern_engine.get_statistics()
            
            output = "# Pattern Detection Statistics\n\n"
            output += f"- **Total Detections**: {stats.get('total_detections', 0)}\n"
            output += f"- **Detection Rate**: {stats.get('detection_rate', 0):.2%}\n\n"
            
            output += "## Detections by Category:\n"
            for category, count in stats.get('by_category', {}).items():
                output += f"- {category.title()}: {count}\n"
            
            output += "\n## Detections by Severity:\n"
            for severity, count in stats.get('by_severity', {}).items():
                output += f"- {severity}: {count}\n"
            
            return output
            
        except Exception as e:
            return f"❌ Error getting statistics: {str(e)}"
    
    def _handle_get_sensitivity_config(self, arguments: Dict[str, Any]) -> str:
        """Get current sensitivity configuration."""
        if not self.pattern_engine:
            return "❌ Pattern detection not available"
        
        try:
            config = self.pattern_engine.get_sensitivity_config()
            
            output = "# Pattern Detection Sensitivity Configuration\n\n"
            output += f"**Global Level**: {config.get('global_level', 'medium')}\n\n"
            
            if config.get('category_overrides'):
                output += "## Category Overrides:\n"
                for category, level in config['category_overrides'].items():
                    output += f"- {category.title()}: {level}\n"
            
            return output
            
        except Exception as e:
            return f"❌ Error getting configuration: {str(e)}"
    
    def _handle_update_sensitivity(self, arguments: Dict[str, Any]) -> str:
        """Update sensitivity configuration."""
        if not self.pattern_engine:
            return "❌ Pattern detection not available"
        
        try:
            global_level = arguments.get('global_level')
            category_overrides = arguments.get('category_overrides', {})
            
            # Update configuration
            if hasattr(self.pattern_engine, 'update_sensitivity'):
                self.pattern_engine.update_sensitivity(
                    global_level=global_level,
                    category_overrides=category_overrides
                )
                return "✅ Sensitivity configuration updated successfully"
            
            return "❌ Sensitivity update not supported by pattern engine"
                
        except Exception as e:
            return f"❌ Error updating sensitivity: {str(e)}"
    
    def _handle_toggle_pattern_detection(self, arguments: Dict[str, Any]) -> str:
        """Toggle pattern detection on/off."""
        enabled = arguments.get('enabled')
        
        if enabled is None:
            return "❌ Missing required parameter: enabled"
        
        try:
            # Update configuration
            if self.credentials:
                if 'pattern_detection' not in self.credentials:
                    self.credentials['pattern_detection'] = {}
                self.credentials['pattern_detection']['enabled'] = enabled
                
                # Save configuration
                # Note: In production, this should persist to the credentials file
                
            return f"✅ Pattern detection {'enabled' if enabled else 'disabled'}"
            
        except Exception as e:
            return f"❌ Error toggling pattern detection: {str(e)}"
    
    def _handle_toggle_category(self, arguments: Dict[str, Any]) -> str:
        """Toggle a specific pattern category."""
        category = arguments.get('category')
        enabled = arguments.get('enabled')
        
        if not category or enabled is None:
            return "❌ Missing required parameters: category, enabled"
        
        if not self.pattern_engine:
            return "❌ Pattern detection not available"
        
        try:
            if hasattr(self.pattern_engine, 'toggle_category'):
                self.pattern_engine.toggle_category(category, enabled)
                return f"✅ Category '{category}' {'enabled' if enabled else 'disabled'}"
            
            return "❌ Category toggle not supported by pattern engine"
                
        except Exception as e:
            return f"❌ Error toggling category: {str(e)}"
    
    def _handle_add_pattern_keywords(self, arguments: Dict[str, Any]) -> str:
        """Add custom keywords to a pattern category."""
        category = arguments.get('category')
        keywords = arguments.get('keywords', [])
        
        if not category or not keywords:
            return "❌ Missing required parameters: category, keywords"
        
        if not self.pattern_engine:
            return "❌ Pattern detection not available"
        
        try:
            if hasattr(self.pattern_engine, 'add_keywords'):
                self.pattern_engine.add_keywords(category, keywords)
                return f"✅ Added {len(keywords)} keywords to '{category}' category"
            
            return "❌ Keyword management not supported by pattern engine"
                
        except Exception as e:
            return f"❌ Error adding keywords: {str(e)}"
    
    def _handle_remove_pattern_keywords(self, arguments: Dict[str, Any]) -> str:
        """Remove keywords from a pattern category."""
        category = arguments.get('category')
        keywords = arguments.get('keywords', [])
        
        if not category or not keywords:
            return "❌ Missing required parameters: category, keywords"
        
        if not self.pattern_engine:
            return "❌ Pattern detection not available"
        
        try:
            if hasattr(self.pattern_engine, 'remove_keywords'):
                self.pattern_engine.remove_keywords(category, keywords)
                return f"✅ Removed {len(keywords)} keywords from '{category}' category"
            
            return "❌ Keyword management not supported by pattern engine"
                
        except Exception as e:
            return f"❌ Error removing keywords: {str(e)}"
    
    def _handle_list_pattern_keywords(self, arguments: Dict[str, Any]) -> str:
        """List all keywords for a pattern category."""
        category = arguments.get('category')
        
        if not category:
            return "❌ Missing required parameter: category"
        
        if not self.pattern_engine:
            return "❌ Pattern detection not available"
        
        try:
            if hasattr(self.pattern_engine, 'get_keywords'):
                keywords = self.pattern_engine.get_keywords(category)
                
                output = f"# Keywords for '{category}' Category\n\n"
                if keywords:
                    for keyword in sorted(keywords):
                        output += f"- {keyword}\n"
                else:
                    output += "No keywords configured for this category."
                
                return output
            
            return "❌ Keyword listing not supported by pattern engine"
                
        except Exception as e:
            return f"❌ Error listing keywords: {str(e)}"
    
    def _handle_force_consultation(self, arguments: Dict[str, Any]) -> str:
        """Force AI consultation regardless of pattern detection."""
        context = arguments.get('context')
        category = arguments.get('category')
        multi_ai = arguments.get('multi_ai', False)
        
        if not context or not category:
            return "❌ Missing required parameters: context, category"
        
        if not self.response_manager:
            return "❌ AI consultation not available"
        
        try:
            # Create synthetic pattern for consultation
            synthetic_pattern = [{
                'category': category,
                'severity': 'HIGH',
                'context': context,
                'line_number': 1,
                'matched_text': context[:100]
            }]
            
            response = self.response_manager.get_response(
                synthetic_pattern,
                context,
                force_multi_ai=multi_ai
            )
            
            return response if response else "❌ No consultation response generated"
            
        except Exception as e:
            return f"❌ Error during forced consultation: {str(e)}"
    
    # AI Consultation Manager handlers
    
    def _handle_ai_consultation_strategy(self, arguments: Dict[str, Any]) -> str:
        """Get recommended AI consultation strategy."""
        if not self.ai_consultation_manager:
            return "❌ AI consultation manager not available"
        
        context = arguments.get('context', '')
        priority = arguments.get('priority', 'accuracy')
        
        if not context:
            return "❌ Missing required parameter: context"
        
        try:
            # First detect patterns
            patterns = []
            if self.pattern_engine:
                patterns = self.pattern_engine.detect_patterns(context)
            
            # Get strategy recommendation
            strategy = self.ai_consultation_manager.get_consultation_strategy(
                patterns,
                priority=priority
            )
            
            output = "# AI Consultation Strategy Recommendation\n\n"
            output += f"**Priority**: {priority}\n"
            output += f"**Recommended Mode**: {strategy.get('mode', 'single')}\n"
            output += f"**Selected AIs**: {', '.join(strategy.get('ais', []))}\n\n"
            
            if strategy.get('reasoning'):
                output += f"## Reasoning\n{strategy['reasoning']}\n"
            
            return output
            
        except Exception as e:
            return f"❌ Error getting consultation strategy: {str(e)}"
    
    def _handle_ai_consultation_metrics(self, arguments: Dict[str, Any]) -> str:
        """Get AI consultation metrics."""
        if not self.ai_consultation_manager:
            return "❌ AI consultation manager not available"
        
        try:
            metrics = self.ai_consultation_manager.get_metrics()
            
            output = "# AI Consultation Metrics\n\n"
            output += f"**Total Consultations**: {metrics.get('total_consultations', 0)}\n"
            output += f"**Success Rate**: {metrics.get('success_rate', 0):.2%}\n"
            output += f"**Average Response Time**: {metrics.get('avg_response_time', 0):.2f}s\n\n"
            
            output += "## By AI Model:\n"
            for ai_name, ai_metrics in metrics.get('by_ai', {}).items():
                output += f"\n### {ai_name.upper()}\n"
                output += f"- Consultations: {ai_metrics.get('count', 0)}\n"
                output += f"- Success Rate: {ai_metrics.get('success_rate', 0):.2%}\n"
                output += f"- Avg Response Time: {ai_metrics.get('avg_time', 0):.2f}s\n"
            
            return output
            
        except Exception as e:
            return f"❌ Error getting metrics: {str(e)}"
    
    def _handle_ai_consultation_audit(self, arguments: Dict[str, Any]) -> str:
        """Get AI consultation audit trail."""
        if not self.ai_consultation_manager:
            return "❌ AI consultation manager not available"
        
        limit = arguments.get('limit', 10)
        pattern_category = arguments.get('pattern_category')
        
        try:
            audit_entries = self.ai_consultation_manager.get_audit_trail(
                limit=limit,
                filter_category=pattern_category
            )
            
            output = "# AI Consultation Audit Trail\n\n"
            
            for i, entry in enumerate(audit_entries, 1):
                output += f"## Consultation #{i}\n"
                output += f"- **Timestamp**: {entry.get('timestamp', 'N/A')}\n"
                output += f"- **Pattern Category**: {entry.get('pattern_category', 'N/A')}\n"
                output += f"- **AI Models**: {', '.join(entry.get('ai_models', []))}\n"
                output += f"- **Mode**: {entry.get('mode', 'N/A')}\n"
                output += f"- **Success**: {'✅' if entry.get('success') else '❌'}\n\n"
            
            return output
            
        except Exception as e:
            return f"❌ Error getting audit trail: {str(e)}"
    
    def _handle_ai_governance_report(self, arguments: Dict[str, Any]) -> str:
        """Export AI governance report."""
        if not self.ai_consultation_manager:
            return "❌ AI consultation manager not available"
        
        try:
            report = self.ai_consultation_manager.export_governance_report()
            
            # Format as markdown report
            output = "# AI Governance and Compliance Report\n\n"
            output += f"**Generated**: {report.get('generated_at', 'N/A')}\n"
            output += f"**Period**: {report.get('period', 'All time')}\n\n"
            
            output += "## Executive Summary\n"
            output += f"- Total AI Consultations: {report.get('total_consultations', 0)}\n"
            output += f"- Compliance Rate: {report.get('compliance_rate', 0):.2%}\n"
            output += f"- Average Decision Time: {report.get('avg_decision_time', 0):.2f}s\n\n"
            
            output += "## AI Usage Distribution\n"
            for ai_name, usage in report.get('ai_usage', {}).items():
                output += f"- {ai_name.upper()}: {usage:.1%}\n"
            
            output += "\n## Pattern Category Analysis\n"
            for category, data in report.get('category_analysis', {}).items():
                output += f"- {category.title()}: {data.get('count', 0)} consultations\n"
            
            return output
            
        except Exception as e:
            return f"❌ Error generating governance report: {str(e)}"