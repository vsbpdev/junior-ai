#!/usr/bin/env python3
"""
Migration script to move credentials from plain JSON to secure storage.

This script helps users migrate their existing credentials.json to one of the
secure storage backends (environment variables, OS keyring, or encrypted file).
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from secure_credentials import (
    SecureCredentialManager,
    CredentialBackend,
    check_credential_security
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def print_security_assessment():
    """Print current credential security status."""
    assessment = check_credential_security()
    
    print("\n=== Credential Security Assessment ===")
    print(f"Available backends: {', '.join(assessment['available_backends'])}")
    print(f"Current backend: {assessment.get('active_backend', 'None')}")
    print(f"Security level: {assessment.get('security_level', 'Unknown')}")
    
    if assessment.get('plain_json_exists'):
        print("\n⚠️  WARNING: Plain text credentials.json found!")
        print("   Your API keys are stored unencrypted.")
    
    if assessment.get('recommendation'):
        print(f"\n💡 Recommendation: {assessment['recommendation']}")


def generate_env_template(credentials: dict) -> str:
    """Generate .env template from existing credentials."""
    lines = ["# Junior AI Assistant Environment Variables", ""]
    
    # Process AI credentials
    for ai_name in ["gemini", "grok", "openai", "deepseek", "openrouter"]:
        if ai_name in credentials:
            ai_config = credentials[ai_name]
            upper_name = ai_name.upper()
            
            lines.append(f"# {ai_name.title()} Configuration")
            
            if "api_key" in ai_config:
                lines.append(f"JUNIOR_AI_{upper_name}_API_KEY={ai_config['api_key']}")
            
            if "enabled" in ai_config:
                lines.append(f"JUNIOR_AI_{upper_name}_ENABLED={'true' if ai_config['enabled'] else 'false'}")
            
            if "model" in ai_config:
                lines.append(f"JUNIOR_AI_{upper_name}_MODEL={ai_config['model']}")
            
            if "base_url" in ai_config:
                lines.append(f"JUNIOR_AI_{upper_name}_BASE_URL={ai_config['base_url']}")
            
            lines.append("")
    
    # Process pattern detection settings
    if "pattern_detection" in credentials:
        pd_config = credentials["pattern_detection"]
        lines.append("# Pattern Detection Configuration")
        
        if "enabled" in pd_config:
            lines.append(f"JUNIOR_AI_PATTERN_ENABLED={'true' if pd_config['enabled'] else 'false'}")
        
        if "sensitivity" in pd_config and "global_level" in pd_config["sensitivity"]:
            lines.append(f"JUNIOR_AI_PATTERN_SENSITIVITY={pd_config['sensitivity']['global_level']}")
        
        lines.append("")
    
    return "\n".join(lines)


def migrate_to_environment(manager: SecureCredentialManager, 
                          output_file: Optional[Path] = None) -> bool:
    """Migrate credentials to environment variables format."""
    credentials = manager.load_credentials()
    if not credentials:
        logger.error("No credentials found to migrate")
        return False
    
    env_content = generate_env_template(credentials)
    
    if output_file:
        # Write to specified file
        output_file.write_text(env_content)
        print(f"\n✅ Generated {output_file}")
        print("\nTo use these credentials:")
        print(f"1. Review and edit {output_file} as needed")
        print("2. Add to your shell profile or use with your deployment")
        print("3. Restart your shell or run: source {output_file}")
    else:
        # Write to .env in current directory
        env_file = Path(".env")
        
        # Check if .env already exists
        if env_file.exists():
            response = input("\n.env file already exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Migration cancelled")
                return False
        
        env_file.write_text(env_content)
        print(f"\n✅ Generated {env_file}")
        print("\nThe application will automatically load this file.")
    
    return True


def migrate_to_keyring(manager: SecureCredentialManager) -> bool:
    """Migrate credentials to OS keyring."""
    if CredentialBackend.KEYRING not in manager.get_available_backends():
        logger.error("Keyring backend is not available on this system")
        logger.info("Install python-keyring: pip install keyring")
        return False
    
    return manager.migrate_from_plain_json(
        target_backend=CredentialBackend.KEYRING,
        delete_original=False
    )


def migrate_to_encrypted(manager: SecureCredentialManager) -> bool:
    """Migrate credentials to encrypted file."""
    if CredentialBackend.ENCRYPTED_FILE not in manager.get_available_backends():
        logger.error("Encrypted file backend is not available")
        logger.info("Install cryptography: pip install cryptography")
        return False
    
    return manager.migrate_from_plain_json(
        target_backend=CredentialBackend.ENCRYPTED_FILE,
        delete_original=False
    )


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Junior AI credentials to secure storage"
    )
    
    parser.add_argument(
        "--target",
        choices=["env", "keyring", "encrypted"],
        default="env",
        help="Target storage backend (default: env)"
    )
    
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Output file for environment variables (default: .env)"
    )
    
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Delete original credentials.json after successful migration"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check security status without migrating"
    )
    
    args = parser.parse_args()
    
    # Print security assessment
    print_security_assessment()
    
    if args.check_only:
        return 0
    
    # Initialize credential manager
    manager = SecureCredentialManager()
    
    # Check if there are credentials to migrate
    if not Path("credentials.json").exists():
        print("\n❌ No credentials.json found to migrate")
        print("\nTo set up credentials securely from scratch:")
        print("1. Copy credentials.template.json to .env and fill in values")
        print("2. Or use the setup.sh script with --secure flag")
        return 1
    
    # Perform migration based on target
    print(f"\n🔄 Migrating to {args.target} backend...")
    
    success = False
    if args.target == "env":
        success = migrate_to_environment(manager, args.env_file)
    elif args.target == "keyring":
        success = migrate_to_keyring(manager)
    elif args.target == "encrypted":
        success = migrate_to_encrypted(manager)
    
    if success:
        print("\n✅ Migration successful!")
        
        if args.delete_original:
            try:
                Path("credentials.json").unlink()
                print("✅ Deleted original credentials.json")
            except Exception as e:
                logger.error(f"Failed to delete original: {e}")
        else:
            print("\n⚠️  Original credentials.json still exists")
            print("   Delete it manually after verifying the migration")
        
        # Update .gitignore if needed
        gitignore = Path(".gitignore")
        if gitignore.exists() and args.target == "env":
            content = gitignore.read_text()
            if ".env" not in content:
                with open(gitignore, 'a') as f:
                    f.write("\n# Secure credentials\n.env\n.env.local\n")
                print("\n✅ Updated .gitignore to exclude .env files")
        
        return 0
    else:
        print("\n❌ Migration failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())