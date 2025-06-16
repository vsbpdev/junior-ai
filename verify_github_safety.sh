#!/bin/bash
# GITHUB SAFETY VERIFICATION SCRIPT
# Run this before ANY GitHub operations

echo "=== GITHUB SAFETY VERIFICATION ==="
echo

# Check 1: Remote URLs
echo "1. Checking Git Remotes:"
git remote -v | grep -v "vsbpdev/junior-ai" && echo "❌ DANGER: Wrong remote found!" || echo "✅ All remotes point to vsbpdev/junior-ai"
echo

# Check 2: Default repo for gh
echo "2. Checking GitHub CLI default:"
DEFAULT_REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null)
if [ "$DEFAULT_REPO" = "vsbpdev/junior-ai" ]; then
    echo "✅ Default repo is correct: $DEFAULT_REPO"
else
    echo "❌ DANGER: Default repo is wrong: $DEFAULT_REPO"
    exit 1
fi
echo

# Check 3: gh-resolved config
echo "3. Checking gh-resolved setting:"
GH_RESOLVED=$(git config --get remote.origin.gh-resolved)
if [ "$GH_RESOLVED" = "base" ]; then
    echo "✅ gh-resolved is set to: base"
else
    echo "❌ WARNING: gh-resolved not set or incorrect"
    echo "   Running: gh repo set-default vsbpdev/junior-ai"
    gh repo set-default vsbpdev/junior-ai
fi
echo

# Check 4: Branch tracking
echo "4. Checking branch tracking:"
git branch -vv | grep -E "RaiAnsar|claude_code-multi-AI-MCP" && echo "❌ DANGER: Branches tracking wrong upstream!" || echo "✅ No branches track wrong upstream"
echo

# Check 5: Fork parent (informational)
echo "5. Fork Information (GitHub API):"
PARENT=$(gh api /repos/vsbpdev/junior-ai --jq '.parent.full_name // "Not a fork"' 2>/dev/null)
echo "   Parent repository: $PARENT"
echo "   ⚠️  This is why we MUST use --repo flag explicitly"
echo

# Final verdict
echo "=== SAFE PR COMMAND ==="
echo "ALWAYS USE: gh pr create --repo vsbpdev/junior-ai"
echo "NEVER USE:  gh pr create"
echo

# Safety check
read -p "Do you understand you MUST use --repo flag? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "❌ Aborted. You must understand the safety requirements."
    exit 1
fi

echo "✅ Verification complete. Safe to proceed with explicit --repo flag."