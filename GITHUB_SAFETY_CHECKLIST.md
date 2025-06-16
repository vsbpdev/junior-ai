# CRITICAL GITHUB SAFETY CHECKLIST

## NEVER EVER SEND PRs TO THE WRONG REPOSITORY

### Before ANY GitHub Operation:

1. **VERIFY DEFAULT REPO**:
   ```bash
   gh repo view --json nameWithOwner --jq .nameWithOwner
   # MUST OUTPUT: vsbpdev/junior-ai
   ```

2. **ALWAYS USE EXPLICIT REPO**:
   ```bash
   # WRONG:
   gh pr create

   # CORRECT:
   gh pr create --repo vsbpdev/junior-ai
   ```

3. **TRIPLE CHECK BEFORE PR**:
   ```bash
   # Check remotes
   git remote -v
   # MUST SHOW: origin https://github.com/vsbpdev/junior-ai.git

   # Check default
   gh repo view --json nameWithOwner --jq .nameWithOwner
   # MUST SHOW: vsbpdev/junior-ai
   ```

### ABSOLUTE RULES:
- NEVER use `gh pr create` without `--repo vsbpdev/junior-ai`
- NEVER trust defaults
- NEVER skip verification
- ALWAYS verify THREE times before any PR

### Fork Warning:
This repository is technically a fork of RaiAnsar/claude_code-multi-AI-MCP.
GitHub's default behavior for forks is to create PRs to the parent.
We've overridden this with `gh repo set-default` but ALWAYS BE EXPLICIT.

### Emergency Recovery:
If a PR is accidentally created to the wrong repo:
1. IMMEDIATELY close it
2. Delete any branch references
3. Verify no data was exposed

### This Repository Belongs To:
**vsbpdev/junior-ai** - NO OTHER REPOSITORY EXISTS FOR THIS PROJECT