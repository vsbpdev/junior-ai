"""Handlers for async cache tool calls."""

from typing import Dict, Any
from .base import BaseHandler


class CacheToolsHandler(BaseHandler):
    """Handles async cache related tool calls."""
    
    def get_tool_names(self) -> list[str]:
        """Return list of tool names this handler supports."""
        return [
            "cache_stats",
            "clear_cache",
            "async_pattern_check"
        ]
    
    def handle(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Handle a cache tool call."""
        if tool_name == "cache_stats":
            return self._handle_cache_stats(arguments)
        elif tool_name == "clear_cache":
            return self._handle_clear_cache(arguments)
        elif tool_name == "async_pattern_check":
            return self._handle_async_pattern_check(arguments)
        else:
            return f"❌ Unknown cache tool: {tool_name}"
    
    def _handle_cache_stats(self, arguments: Dict[str, Any]) -> str:
        """Get async pattern cache statistics."""
        if not self.async_pipeline:
            return "❌ Async cache not available"
        
        try:
            # Get cache stats
            if hasattr(self.async_pipeline, 'get_cache_stats'):
                stats = self.async_pipeline.get_cache_stats()
            else:
                return "❌ Cache statistics not available"
            
            output = "# Async Pattern Cache Statistics\n\n"
            output += f"**Cache Size**: {stats.get('size', 0)} entries\n"
            output += f"**Max Size**: {stats.get('max_size', 0)} entries\n"
            output += f"**Hit Rate**: {stats.get('hit_rate', 0):.2%}\n"
            output += f"**Total Hits**: {stats.get('hits', 0)}\n"
            output += f"**Total Misses**: {stats.get('misses', 0)}\n"
            output += f"**Memory Usage**: {stats.get('memory_bytes', 0) / 1024 / 1024:.2f} MB\n\n"
            
            # Deduplication stats
            if 'deduplication' in stats:
                output += "## Deduplication Statistics\n"
                dedup = stats['deduplication']
                output += f"- Duplicate Requests Avoided: {dedup.get('duplicates_avoided', 0)}\n"
                output += f"- Time Saved: {dedup.get('time_saved_ms', 0) / 1000:.2f}s\n"
                output += f"- Current In-Flight: {dedup.get('in_flight', 0)}\n\n"
            
            # TTL stats
            if 'ttl' in stats:
                output += "## TTL Statistics\n"
                ttl = stats['ttl']
                output += f"- Expired Entries: {ttl.get('expired', 0)}\n"
                output += f"- Average TTL: {ttl.get('avg_ttl', 0):.1f}s\n"
                output += f"- Oldest Entry: {ttl.get('oldest_age', 0):.1f}s ago\n"
            
            return output
            
        except Exception as e:
            return f"❌ Error getting cache statistics: {str(e)}"
    
    def _handle_clear_cache(self, arguments: Dict[str, Any]) -> str:
        """Clear the async pattern cache."""
        confirm = arguments.get('confirm', False)
        
        if not confirm:
            return "⚠️ Cache clearing requires confirmation. Set 'confirm' to true to proceed."
        
        if not self.async_pipeline:
            return "❌ Async cache not available"
        
        try:
            # Clear cache
            if hasattr(self.async_pipeline, 'clear_cache'):
                cleared = self.async_pipeline.clear_cache()
                return f"✅ Cache cleared successfully. Removed {cleared} entries."
            else:
                return "❌ Cache clearing not supported"
                
        except Exception as e:
            return f"❌ Error clearing cache: {str(e)}"
    
    def _handle_async_pattern_check(self, arguments: Dict[str, Any]) -> str:
        """Perform async pattern detection with caching."""
        text = arguments.get('text', '')
        sensitivity_level = arguments.get('sensitivity_level', 'medium')
        auto_consult = arguments.get('auto_consult', True)
        
        if not text:
            return "❌ Missing required parameter: text"
        
        if not self.async_pipeline:
            return "❌ Async pattern detection not available"
        
        try:
            # Perform async pattern detection
            import asyncio
            
            async def detect_async():
                return await self.async_pipeline.detect_patterns_async(
                    text,
                    sensitivity_level=sensitivity_level
                )
            
            # Run async operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(detect_async())
            finally:
                loop.close()
            
            # Check if result was from cache
            from_cache = result.get('from_cache', False)
            patterns = result.get('patterns', [])
            
            # Format output
            output = "# Async Pattern Detection Results\n\n"
            
            if from_cache:
                output += "ℹ️ **Result retrieved from cache**\n\n"
            
            if not patterns:
                output += "✅ No significant patterns detected"
            else:
                for pattern in patterns:
                    output += f"## {pattern['category'].title()} Pattern Detected\n"
                    output += f"- **Severity**: {pattern['severity']}\n"
                    output += f"- **Context**: {pattern['context'][:200]}...\n"
                    output += f"- **Line**: {pattern.get('line_number', 'N/A')}\n\n"
            
            # Auto-consult if requested and patterns found
            if auto_consult and patterns and self.response_manager:
                consultation = self.response_manager.get_response(patterns, text)
                if consultation:
                    output += "\n## AI Consultation\n\n"
                    output += consultation
            
            # Add performance metrics
            if 'metrics' in result:
                metrics = result['metrics']
                output += f"\n---\n*Detection time: {metrics.get('time_ms', 0):.1f}ms"
                if from_cache:
                    output += " (from cache)"
                output += "*"
            
            return output
            
        except Exception as e:
            return f"❌ Error during async pattern detection: {str(e)}"