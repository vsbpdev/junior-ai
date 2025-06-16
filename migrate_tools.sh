#!/usr/bin/env bash
# Migration helper script for updating tool names from multi-ai-collab to junior-ai

# Exit on error, undefined variables, and pipe failures
set -euo pipefail

echo "🔄 Junior AI Assistant Migration Helper"
echo "This script helps update your files from old tool names to new ones."
echo ""

# Check if user wants to proceed
read -p "This will update all .md, .txt, and .sh files in the current directory. Continue? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Migration cancelled."
    exit 1
fi

# Count files that will be updated (excluding .git directory)
count=$(grep -r "mcp__multi-ai-collab__" . --exclude-dir=.git --include="*.md" --include="*.txt" --include="*.sh" 2>/dev/null | wc -l | tr -d ' ')

if [ "$count" -eq "0" ]; then
    echo "✅ No files need updating. You're already using the new tool names!"
    exit 0
fi

echo "Found $count instances to update..."

# Perform the replacement
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    find . -type f \( -name "*.md" -o -name "*.txt" -o -name "*.sh" \) -not -path "./.git/*" -exec sed -i '' 's/mcp__multi-ai-collab__/mcp__junior-ai__/g' {} +
else
    # Linux
    find . -type f \( -name "*.md" -o -name "*.txt" -o -name "*.sh" \) -not -path "./.git/*" -exec sed -i 's/mcp__multi-ai-collab__/mcp__junior-ai__/g' {} +
fi

echo "✅ Migration complete! All tool names have been updated."
echo ""
echo "Don't forget to:"
echo "1. Remove the old server: claude mcp remove multi-ai-collab"
echo "2. Install the new server: ./setup.sh"
echo "3. Test with: mcp__junior-ai__server_status"